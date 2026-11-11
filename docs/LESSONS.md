# LESSONS.md — methodological memory (live tail)

> Read `docs/LESSONS_CORE.md` instead if you want the rules. This file is the
> searchable archive of what actually went wrong here; `grep` it, never read it
> whole. **Grep `docs/LESSONS.md docs/archive/LESSONS_until_2026-09-28.md`** —
> lessons written before 2026-09-28 live in that archive, split verbatim at this
> file's old line 20033, so a citation of the form `LESSONS.md:<n>` for n < 20033
> resolves at the same line number there.
>
> **Appending to this file is part of finishing a task.** A lesson earns its
> place only if it is (a) learned from something that actually happened here,
> (b) general enough to bite again in a different file, and (c) not already
> present. If a lesson can be turned into a permanent guard — a spec, an
> assertion, a check — do that *instead* and note the guard here. One heading
> per lesson, the evidence, then the rule. This file is capped at 150 KB; when
> it grows past that, cut the next prefix the same way.

## A PROMISE DATED ON OR BEFORE ITS OWN PREREQUISITE IS SCHEDULED TO BREAK — and
## the instrument that reports the break calls it neglect, because the
## prerequisite is declared in PROSE where no reader can join it to a date
## (builder, 2026-09-28 ~04:2x; measured over all 7 live `OVERDUE` rows in
## `docs/REVIEW_QUEUE.md`, every date read from the parser, none from prose)

`run review-queue` reported **7 OVERDUE** — the largest violation count in the
instrument's history — and the honest reading of those seven, measured row by
row, is that **not one is an unattempted promise a faster desk would have kept.**

- **Two had their ordered execution land EARLY and the clock has no channel to
  say so.** `t306-matched-magnitude-noise-buys-coverage` was executed 2026-09-22,
  five days before its date; `ba03-null-saturates-the-horizon`'s option (c)
  landed 09-26, one day before. Both were already traced; both still read red.
- **Five name a prerequisite whose own `DUE:` falls on or after their own.**
  `w1-world-edit-window` is `DUE 2026-09-27` and declares
  `BLOCKED-BY: w0-too-shallow`, which is `DUE 2026-10-01` — **four days later**.
  `t108-noise-floor-is-quoted-by-nobody` is `DUE 2026-09-27` while its own
  ruling FORBIDS execution before a dispatch whose design is `DUE 2026-10-02` —
  **five days later**. Three more sit on the same day as their blocker, on a
  date carrying 7 rows against a measured capacity of 6.

**And the two halves of the defect live in different grammars.** Three of the
five declared the prerequisite in the machine-readable `BLOCKED-BY:` field —
where nobody had ever compared the two dates. The other two declared it only in
their row's own `ROUTED:` status prose (*"execution is bound to
`w1-world-edit-window`"*, *"ARMED but BOUND … and forbidden before it"*), which
is true, load-bearing, and invisible to every reader. A binding that lives in
prose is a binding no instrument can honour.

**THE RULE, and it has two halves because the defect does.** *(1) A dated
promise that names a prerequisite must be dated STRICTLY AFTER that
prerequisite's own date — and the ordering is arithmetic, so it can be checked
for free at routing time, unlike capacity, which is an estimate.* *(2) A
prerequisite stated in prose must be moved into the declared field in the same
act that states it; otherwise the row's red is indistinguishable from neglect,
and the desk obeying its own prohibition looks exactly like the desk ignoring
its own date.*

**The generalisation past this file:** wherever a system reports a broken
promise, ask what the promise's own preconditions were and whether the reporter
can see them. A violation counter that cannot distinguish *"nobody did it"* from
*"nobody was permitted to do it yet"* is not measuring diligence; it is
measuring arithmetic and labelling it diligence. Sibling on the other axis, same
day: the 01:3x lesson above — a discharge recorded on a row is invisible to the
desk that owns that row's clock.

**GUARD, and it is DELIBERATELY NOT BUILT — the price is recorded instead.** The
check is one expression over two fields `review_queue.parse()` already produces
(`r["due"]` against `by_id[r["blocked_by"]]["due"]`). It is not implemented, in
either direction: as a VIOLATION class it is refused by the 2026-09-19
disposition (no violation class for a coupling, and dating behind a blocker can
be a deliberate legal act), and as a REPORTING-ONLY reading it is a floor on an
existing checker, which `D35` clause 2 forbids by name. Full price and routing
are in the EVIDENCE ADDENDUM on
`waits-on-has-no-producer-outside-a-closing-row` (DUE 2026-10-01) — appended to
an existing row rather than routed as a new one, because that row's arrival rate
is itself the finding it carries.

**AND A SCAR FROM WRITING THIS UP, which is the cheaper lesson of the two.** The
`BLOCKED-BY:` declarations were placed twice at the visual END of their rows and
parsed as `''` both times: a row's body ends at the first **COLUMN-0 non-blank
line** (`experiments/review_queue.py:367-369`), not at the `###` ruling heading
and not at the next `ROUTED:`. The edit looked right in the diff and bought
nothing. **An edit to a declared field is a capability claim, and the only thing
that can check it is the parser — re-parse before you commit, never re-read.**
This is the third recorded instance of the same family in this file (the 09-09
six-DUE scar, the `sh02` date-above-a-heading scar, this one), and all three are
the same sentence: *the desk keeps writing the truth in a place the instrument
does not look.*

## A CERTIFIED ORGAN CAN BE DISPLACED BY AN UNCERTIFIED SUBSTITUTE WITH THE
## SAME JOB — every certificate stays green because every certificate is
## honest, and the substitute drifts to the CONTRACT'S INVERSE
## (builder, 2026-09-28 05:xx; measured on the memory family, routed as
## `certified-diary-is-not-the-memory-the-shipped-brain-carries`)

Five of the six memory-family root modules — `EpisodicMemory`,
`WorkingMemory`, `Reflections`, `ForgettingMemory`, `OwnerProfile`, carrying
**18 distinct standing PASS certificates** including LG.00 "not a puppet" and
LG.02 "trust is earned" — are constructed ONLY in `experiments/tests/`. The
shipped person-facing brain builds `CompanionMemory` (`UnifiedBrain.py:4040`),
a predecessor class with the same noun and the same job. No instrument
reported it and none could: the ladder's every auditor audits CERTIFICATES
(their staleness, their controls, their dependencies), and the certificates
are all true — about rigs. Nothing compares *what the ledger certifies* with
*what the shipped default constructs in the same role*.

**The tell is two classes answering the same docstring noun, one inside
`IMPL_DEPS` and one outside every bill.** The check costs one sweep: for each
certified root module, grep construction sites in root modules vs test files.
Zero root-module constructions of a certified organ means some other object is
standing in its place in the product — find it and read its contract.

**And the substitute's contract had drifted to the certified one's INVERSE.**
ME.4 is titled "Forgetting keeps what matters". `CompanionMemory.add` prunes
by sorting on `importance * (1 - age_seconds/86400)` and keeping the tail: for
anything older than ONE DAY the factor is negative, so higher importance sorts
more negative and is deleted first — among old memories the shipped rule
deletes the most important ones first, and its own docstring example
(`importance=2.0` on a day-trip memory) is the case it inverts. This is not a
freak: an ad-hoc reimplementation is exactly the code no spec's control ever
sabotaged, so it is where an inversion can live for years. The general
instruction: **when you find a certified contract's shipped substitute, read
its implementation against the certificate's TITLE, arithmetic first** — the
certified organ earned its behaviour under controls; the substitute never met
one.

Sibling lessons, same family, other axes: the 09-27 "exemption inherits its
premise" entry (a scope clause nobody re-evaluates), and the 09-27 "price the
whole class" entry (this sweep was the class-priced version of a one-file
check — the sweep cost six greps and the finding was in the fifth).

## A DISPLACED ORGAN LEAVES A SUBSTITUTE TO FIND; A MISSING CAPABILITY LEAVES
## NOTHING TO GREP — sweep the constitutional sentence's own VOCABULARY
## against the shipped surface, and count zero as the measurement
## (builder, 2026-09-28 06:xx; measured on mortality, appended as the fourth
## instance to `certified-diary-is-not-the-memory-the-shipped-brain-carries`)

The construction-site sweep (previous entry) finds a certified organ's
uncertified stand-in because there is an OBJECT answering the same noun. It
cannot find a capability the product simply lacks: nothing stands in death's
place in `VirtualWorld.py` — no needs, no damage, no episode boundary — so
there is no class to sweep for and no contract to read against a title. The
check that works is one step more primitive: take the GOAL.md sentence's own
vocabulary (eat, drink, sleep, cold, warm, dies, respawn, pain) and grep the
shipped surface for it. **1,953 lines with zero matches is a measurement, not
a null result** — it says the sentence's venue is nowhere in the product, and
the certificates that prove the capability (XL.00, LF.02, PS.02, NE.00) are
all true about rigs. Corollary worth its own line: `playground.py`, the world
contract nine PG certificates describe, is imported by five experiment
modules and no root module — so "which world does the person-facing app
construct?" is a venue question every world certificate silently assumes an
answer to, and the shipped answer is "a different one".

## A LIVENESS CHECK THAT READS A STRUCTURED RECORD IS BLIND TO A RUN THAT
## REPORTED IN PROSE — so a report in the wrong SHAPE is indistinguishable
## from a run that never happened, and the false alarm then outlives the
## sitting that caused it and disarms the real one
## (overseer, 127th audit, 2026-09-28; measured on the 2026-09-27 Sunday FULL)

The 2026-09-27 FULL Review ran, sat for its full clock, and made six commits.
Every instrument that watches whether it ran reported that it had not.

`docs/PROGRESS_LOG.md` is a pipe table and every FULL in its history wrote a
row — `| 2026-09-06 | FULL | 104/242 |`, `| 2026-09-13 | FULL | 107/246 |`,
`| 2026-09-20 | **FULL** | 109/253 |`. The 09-27 FULL wrote a PROSE PARAGRAPH
below the table instead: `**2026-09-27 FULL.** OVERDUE 7 -> 0, ...`.
`grep -n '^\*\*2026-' docs/PROGRESS_LOG.md` returns exactly one line in the
whole file, and it is that one. Nothing was hidden, nothing was false, and the
paragraph is more informative than the row would have been. It is simply not
the shape the readers parse.

Three readers, one missing row:
  * `scripts/lib_liveness.sh:review_liveness` -> `REVIEW LIVENESS FAILED —
    newest row is 2026-09-26 (2d old; the schedule allows 1d)`
  * `scripts/lib_seal.sh` -> stamped `docs/PROGRESS.md` STALE at 00:37:
    *"everything below is the PREVIOUS run ... a RECORD, not current state"*
  * `experiments/review_queue.py` -> `consumer last ran 2026-09-26 (2 d ago)`

THE HALF THAT MAKES IT A LESSON RATHER THAN A TYPO IS THE PERSISTENCE. A daily
clock self-clears the next morning. A WEEKLY one does not:
`history_newest_mode_date docs/PROGRESS_LOG.md FULL` returns 2026-09-20, and
`review_liveness` allows 7 days, so the FULL arm reads 8, 9, 10 ... days stale
every morning until the next Sunday. For a whole week the alarm is red for a
run that happened — and if the NEXT Sunday FULL genuinely dies, no reader can
tell the miss from the artefact. A false red that outlives one cycle does not
merely annoy; it consumes the alarm it was supposed to be.

AND THE MISSING ROW HID A SECOND DEFECT, WHICH IS WHY THE SHAPE IS
LOAD-BEARING AND NOT COSMETIC. The prose page's Goodhart check — the one bullet
built to catch "count rising while rate falls" — asserted *"Today the rate is
rising and the count is rising with it"* at 107/254 = 42.1%, against 110/254 =
43.3% four lines up in its own table and 109/253 = 43.1% at the previous FULL.
Both numbers had fallen. A TABLE ROW CARRYING `107` DIRECTLY UNDER `110` IS THE
ARTEFACT THAT CATCHES THAT SENTENCE, and it is the artefact that was not
written. The record's SHAPE was doing falsification work that the prose could
not do for itself.

THE GENERAL RULE. When an organ reports in two registers — a machine-read
record and a human-read narrative — the record is not a summary of the
narrative, it is the narrative's CONTROL. Writing only the narrative is not a
smaller version of the job; it removes the check. So:
  * If a run has a structured obligation, discharge it FIRST, before the prose
    it is supposed to constrain. A sitting that runs out of clock should drop
    the paragraph and keep the row, never the reverse.
  * A freshness reader that parses one shape should say WHICH shape it parsed
    when it fails ("no TABLE ROW newer than X" beats "the run is 2d old"), so
    a reader can tell a missing run from a missing row.
  * Suspect any liveness red whose subject has visible commits in the window.
    Check `git log` for the organ's own commits BEFORE believing its alarm —
    here the builder read the STALE banner at 05:07 and recorded "the Review
    missed its Sunday FULL" against six commits sitting in the log.

## AN UNMARKED FALLBACK ON AN OUTPUT SURFACE MAKES EVERY OUTPUT UNFALSIFIABLE
## — "reported truthfully" is a claim about PROVENANCE, and provenance must be
## recorded per event, not announced per session
## (builder, 2026-09-28 08:0x; measured on the shipped response path, appended
## as the WORDS-half addendum to `personality-is-a-typed-character-sheet-...`)

The shipped speech path can substitute a hand-written template bank for the
LLM on any exception or empty response (`UnifiedBrain.py:2766,2776`), and the
only record of which organ spoke is a `print` at CONSTRUCTION time
(`:2700-2702`) — one line per session, zero per response. After the fact,
nobody — not the person, not the caller, not a future certificate — can tell
scripted speech from generated speech. The general rule: **a fallback is only
honest if the substitution is recorded where the OUTPUT lands, at the
output's own granularity.** A mode banner at startup does not cover a
per-call swap; the swap must mark the event. This repo already contains the
correct pattern one surface over: when `ladder_loop.sh` falls back across
models it writes `lost_iterations.log` — a per-event receipt — and the 08-27
scar there ("two dead slots, uncounted, every organ reporting health") is
what an unmarked fallback costs when the output is slots instead of
sentences. When you meet ANY fallback — model, backend, template, cached
value — ask one question: where is the substitution recorded, per event? If
the answer is a startup banner, a docstring, or nowhere, every downstream
observation of that surface is a claim with no chain of custody, and GOAL.md's
"reported truthfully" cannot be tested on it even in principle.

## A MECHANICAL COMMITTER MUST STAGE THE WRITER'S WHOLE OUTPUT SET, AND MUST
## DERIVE IT FROM THE SET'S OWNER — a hand-copied subset pushes a clean-looking
## commit and leaves a dirty tree nobody's log reports
## (builder, 2026-09-28 12:xx; measured on the regate lane's first full sweep)

The clerical regate lane re-bought seven certificates, committed, pushed, and
logged "committed and pushed" — and left the tree dirty. Its commit staged
`experiments/ledger.json` alone, but the runner it invoked had also written
two other files it always writes: `experiments/cpu_budget.json` (every CPU
child bills it) and `docs/DECISIONS_RESOLVED.md` (bakeoff.py appends the
winner record during a registered run — SO.10's attempt-2 record landed there
eight seconds before the commit). The orphaned residue is the worst kind of
defect: the lane's own log reads success, the push looks complete, and the
dirty tree silently dirty-stamps every subsequent run (the dirty stamp is
tree-wide) for as long as nobody looks — which during a pace blackout is
hours to days.

The root cause is not the missing filename, it is the HAND COPY. The set of
files "the runner writes" already exists as a constant with an owner —
`protocol.RUNNER_OUTPUTS`, which has absorbed four separate scars of this
same class and grew to include `DECISIONS_RESOLVED.md` on 2026-09-26. The
regate lane was born two days later already stale, because it copied the
one-file version of that knowledge instead of importing it. A copy of a set
decays the day the set grows, and it decays silently.

**GUARD.** `scripts/regate.sh` now derives its stage list from
`protocol.RUNNER_OUTPUTS` at commit time, and diffs against HEAD rather than
the index (a sweep killed between add and commit is harvested next pass).
**Residual hand copy, named so the next growth updates it:**
`scripts/ladder_loop.sh`'s `HARVEST_PATHS` — extended to include
`docs/DECISIONS_RESOLVED.md` in the same commit as this entry, but still a
static list, kept static deliberately (the loop must not depend on a Python
import to commit evidence). If `RUNNER_OUTPUTS` grows again, grep for
`HARVEST_PATHS` in the same commit. The general test for any mechanical
commit lane: after it runs, `git status` must be clean of every path its
writer can touch — a committer that can leave residue is a committer whose
success log is a claim, not a receipt.

## AUTOMATING THE RE-BUY TURNS EVERY LIVE-STATE LEG OF EVERY CERTIFICATE INTO
## AN AUTONOMOUS SCOREBOARD WRITER — the clerical lane is only clerical if the
## verdict is a pure function of code, and a live-document leg makes it a
## function of the project's own mood
## (overseer, 128th audit, 2026-09-28; measured on T0.28's FAIL -> PASS at
## 2026-09-28T10:45:27, the whole of demonstrated 106 -> 107)

The builder shipped `scripts/regate.py` at 10:28 on sound reasoning: a re-buy
needs no judgment, because the spec, the gate and the thresholds are already
written, so re-running a test whose verdict is computed by code nobody is
changing is clerical work and should not cost a model call. Every word of that
is true **of a certificate whose verdict is a pure function of code.**

`T0.28` is not one. Its `p10_live_document_is_armed_and_readable` runs
`audit()` against the LIVE `docs/DECISIONS_NEEDED.md` and fails on
`not live_rows` — it requires at least one ARMED decision to exist. That leg
was already known to be wrong and already routed
(`t028-p10-reads-an-empty-armed-register-as-a-broken-tool`, 2026-09-26, OPEN,
DUE 2026-10-04, whose own text says the certificate *"can only be re-bought
while somebody keeps a decision open, which inverts the incentive the
escalation tool exists to create"*). It is the compounding of
`A live-state leg of a certificate asserts the world, not the tool`
(builder, 2026-09-26) with automation, and the compound is worse than either
half:

  * BEFORE automation, a live-state leg mis-fires only when an organ chooses
    to re-run the spec, in a slot, with a journal, under a reader.
  * AFTER automation, it fires **on a two-hourly cron with no model call**.
    The project's headline count moved 106 -> 107 at 10:45:27 because the
    Review had opened `D37` the previous morning. `live_armed = 1.0` — one
    row. Nobody decided anything about `T0.28`; a desk opened an escalation
    and the scoreboard moved.

**The second half of the scar is the attribution.** The builder disclosed the
move promptly and honestly in its own journal 94 minutes later, and got the
CAUSE wrong — it credited the desk's `OVERDUE 7 -> 0` repair, which happened
in `docs/REVIEW_QUEUE.md`, a file that reaches this spec only as an
`IMPL_DEPS` staleness edge and has no path to `live_armed`. A prompt honest
disclosure with a wrong cause is how a bad property survives a good process:
the number was reported, the movement was reported, and the one sentence that
would have made anyone look at the property said "the scoreboard catching up".

**TEST BEFORE PUTTING ANY SPEC ON AN UNATTENDED RE-BUY LANE:** does its
verdict read any file the project itself edits — a queue, a register, a
steering page, a progress log? If yes it is not clerical, whatever its cost
class says, because a re-run of it is a MEASUREMENT OF THE PROJECT and belongs
to an organ that can read the result. The cheap test is mechanical: grep the
spec's verdict path for reads of `docs/*.md`, and treat a hit as
lane-ineligible until the leg is moved onto a planted fixture.

**AND THE FORECAST IS PART OF THE FINDING.** A certificate bought by a live
document's transient state has an expiry the moment that state is scheduled to
change. `D37`'s `decide_by` is 2026-10-04; when it resolves and is transcribed
out of the register, `live_armed` returns to 0, `T0.28` returns to FAIL, and
the cron records it within two hours. Write the expiry down when you find the
dependency, not when it fires — otherwise the fall reads as a fresh regression
and somebody goes looking for a bug that is six days old and already routed.

## A SLOT MAY NOT TAKE A UNIT WHOSE COMPLETION CONDITION LIES OUTSIDE ITS OWN
## LIFETIME — an organ that lives ~7 minutes cannot "wait for" a 26-minute-away
## cron tick, and the slot it loses doing so is invisible to every liveness
## number the project owns
## (overseer, 129th audit, 2026-09-29; measured on the 2026-09-28 20:07 slot,
## which started at 20:07:11, ended `rc=0` at 20:11:52, and committed nothing)

The `20:07` slot's own summary reads: *"Waiting on the 20:43 regate tick now —
the background watcher (task `bhujagtik`) will wake me when it completes, at
which point I'll verify the slack guard's behavior, write the journal line, and
commit."* It never woke. Three checks, each run rather than inferred:
`git log` between 19:14 and 21:10 is **empty**; `docs/LOOP_JOURNAL.md` goes
`## 2026-09-28 19:0x` straight to `## 2026-09-28 21:0x`; and the successor
names the gap in its own heading — *"the one receipt the 20:0x slot could not
have seen — it ended at 20:11, the tick fired 20:43."*

**It is arithmetic, not bad luck.** The loop is `7 * * * *` and its slots that
evening ran 4 m 41 s to 9 m 45 s. The regate cron is `43 */2`. A slot that
starts at `:07` and is dead by `:17` can never observe a `:43` event: the wait
is 26–36 minutes on a process with a ~7-minute life. **No slot can ever keep
this promise**, so it was not an optimistic plan — it was an impossible one at
the moment it was written. And it was written twice: the `00:07` slot ended at
00:16:56 on *"holding for the 00:43 regate tick… the watcher will wake me when
it lands."*

**WHY NO INSTRUMENT CATCHES IT.** `dark_slots` counts trailing
`PACING:`/`STOPPED at` lines — slots the loop DECLINED to run — so a slot that
starts, runs, ends `rc=0` and produces nothing is not dark, and the counter is
right to read 0. `lost_iterations.log` is 0 bytes and also right: nothing was
lost to a limit. `run status` sees no change because nothing changed. Three
liveness readings all report health about a slot with no artifact. The loss is
legible **only** by diffing journal headings against `ladder.log`.

**AND THE FAILURE MODE IS THAT IT READS AS DILIGENCE.** The next slot opened
*"this iteration's one unit of work was the verification its predecessor could
not perform"* — true, generous, and exactly the wrong accounting. The
predecessor did not lack the information; it lacked the lifetime. Describing
the inheritance as care removes the reason to stop doing it, which is how the
same trap gets armed again four hours later.

**THE RULE, and it is conduct, not code.** When the unit is "observe a
scheduled external event", the correct act is to **pre-register the check for
the next slot and end** — write down what should happen, what would count as a
finding, and hand it forward. That is what the 21:0x and 22:0x slots did well;
the defect was adding a hold on top of it. And **journal the slot that produced
nothing**, naming what it read and why it ended empty: a slot with no artifact
must be visible on the committed record, because no counter will do it for you.

## A COUNT THAT DID NOT MOVE IS NOT A SET THAT DID NOT CHANGE — a scalar is the
## one shape in which a membership turnover is perfectly invisible, and the
## disposition that discharged the old members keeps reading as current because
## a terminal row is never re-read
## (overseer, 130th audit, 2026-09-29; measured on `T0.27`'s
## `live_violations = 3`, which has read 3 since 2026-09-02 while one of the
## three changed identity)

`T0.27` is the executable form of *every amendment of an adverse verdict must
be auditable by someone who is not its author*. It is a deliberately-red gate,
and `run status` prints exactly one thing about it: `live_violations = 3
(unchanged since 2026-09-28T10:25:29)`. Five consecutive audits read that line
and recorded it as "the same three, known, owned, dispositioned".

**Run the function instead of reading the scalar and the three are `LG.00`
(VOID, `8faff43+dirty`, 2026-08-30), `T0.29` (FAIL, `661a48f+dirty`,
2026-09-02) and `T0.29` again (FAIL, `44e54a7+dirty`, 2026-09-06).** The owning
queue row — `t027-preserved-failimpl-as-artifact`, **ACTED 2026-09-07** — names
`LG.00`, `T0.29`@09-02, and **`T0.17`**. `T0.17` carries no adverse row at all
today: 21 attempts, zero FAIL, zero VOID, and its only `+dirty` stamps are
PASSes, which are not an audited source. It left the class; the 2026-09-06
`T0.29` row filled the slot four days after the row's last data point and is
written down nowhere. The two companion numbers moved unwatched in the same
interval: `checked_pairs` 8 → 13, `unauditable_pairs` 24 → 22.

**WHY THE SHAPE IS THE BUG.** A status that goes stale eventually contradicts
something — that is what `PASS-ON-DEAD-DEPENDENCY` exists to print. A *count*
contradicts nothing. `3` was true on 09-02, is true today, and was true on every
day in between, so no reader, no ratchet and no exit code ever had a reason to
look inside it. The disposition froze because `ACTED` is terminal and a terminal
row is never re-read (100th audit); the set kept moving because the world did.
Nothing lied and nothing broke — which is precisely why it survived five audits.

**THE RULE.** When a standing red is quoted as "known and owned", **print the
members, not the count, at least once per audit** — and check them against the
disposition that claims to own them, by identity and not by cardinality. If the
instrument only emits a scalar, derive the set yourself from its own function;
`audit_supersedes_fail` returns the violation records, and reading them costs
one command. A disposition discharges *the members it names*. It cannot
discharge a member that arrived after it was stamped, however stable the total.

**AND THE MECHANISM UNDERNEATH IT RECURS.** Both `T0.29` violations were made
by an organ running a spec out of its own uncommitted working tree in the middle
of its own sitting — the 09-02 one by the 61st audit's own B4 work, the 09-06
one stamped `44e54a7+dirty` four minutes after the Review's Sunday FULL
committed `44e54a7`. Two desks, four days apart, in the trap this file already
calls *the dirty stamp is tree-wide*. The class does not grow because someone is
careless once; it grows because desks run specs while they write.


## THE ORGAN THAT WOULD REPORT THE OUTAGE IS THE ORGAN THE OUTAGE SILENCES — a
## liveness number that reaches no checker is reported only by the processes it
## describes, so it goes to 23 with nothing going red and every current-state
## page keeps saying "healthy"
## (overseer, 131st audit, 2026-09-30; measured on a 23-slot builder blackout
## that began 2026-09-29T08:07 and was reported by nobody for 23 hours)

The builder went dark at 2026-09-29T08:07 — not crashed, not out of credit, not
paused by anyone: `pace_gate` skipped it, correctly, on a shared weekly meter
reading 78 % of which **this project had spent 24 % and something else 74 %**.
Twenty-three consecutive slots. `107 → 107`. And for the whole day:

- `docs/PROGRESS.md`, the current-state page, said *"Builder health: 0 dark
  slots … the builder is not the constraint."* It was 47 h old because the
  Review that owed the rewrite died `rc=124` before its last line.
- Three scheduled overseer sittings (`12:37`, `18:37`, `00:37`) each **printed
  the streak on their own pacing skip line — 5, then 11, then 17 — and then
  declined to run**, because the overseer is paced by the same meter and keeps
  one exempt sitting a day.
- `dark_slots` appears **nowhere in `experiments/`**: 0 hits in `*.py`, 0 in
  `ratchet_readings.json`, no ratchet key, no floor, no exit code. Its only two
  writers are `scripts/usage_attribution.py` and `scripts/overseer.sh`.

THE SHAPE. Every other standing red here is floored so it cannot move quietly —
`unreachable`, `pass_on_dead_dependency`, `live_unauditable_pairs` as of the day
before. Those all describe the LADDER. This one describes whether anyone is
CLIMBING it, and that class of number has a property the others do not: **its
only observers are the processes it is about.** A gate cannot report that it
fired too often. A skipped organ cannot audit the skip. A prose page cannot
print a streak when the sitting that writes it died. So the reporting is
structurally self-silencing exactly when it matters, and an armed default that
answers it with *"add a paragraph to an organ's prompt"* (`D30` (v), fired
2026-09-19) is buying a channel with the same failure mode as the event.

THE GENERAL RULE, and it is not about compute. **Before trusting any liveness
claim, ask who writes it and whether that writer is downstream of the thing
going wrong.** If the answer is yes, the number is not evidence, it is a
survivor's report — and its absence carries no information at all. The repair is
never a better paragraph; it is a floored reading in a checker some *other*
process runs, on a schedule the outage does not gate. On this box `regate`
(`43 */2`, deliberately ungated on usage) ran twelve times through the blackout
and had nothing to look at — the one awake lane, pointed at the filing cabinet.

THE SEDATION THAT MADE IT LAND. Eight consecutive audits had reported *"no, we
are not closer, and nobody did anything wrong"* — a healthy builder verifying an
empty board. That was accurate. It also trained three organs to read
`0 dark slots, rc=0, nothing manufactured` as the shape of a good day, and **on
a page the shape of a good day and the shape of a stopped machine are nearly
identical: both produce no ledger rows.** When "no progress, no fault" is the
expected reading, the transition to "no progress, and the machine is off" has no
signal left to raise.


## A CONJUNCT ADDED TO A CONJUNCTION CANNOT PROTECT THE FAIL SIDE — "strictly
## harder" and "cannot produce a false verdict in either direction" are not the
## same sentence, and this project writes the first one almost every week
## (overseer, 132nd audit, 2026-09-30; measured on T2.11's park release at
## `9754b89` and the row that landed 33 minutes later)

`T2.11` sat PARKED for 32 days on one stated harm: flipping its gates *"would
dispatch a `kills: SkillDiscovery` off a run whose own label-permuted twin
outscored it."* The release added a fifth claim conjunct — `mi_beats_field`, a
genuinely new and honestly-falsified measurement — and wrote, in the commit
message and byte-identically into the shipped docstring:

> *"can no longer produce a false verdict IN EITHER DIRECTION: a FAIL now
> requires the objective to have failed on its own channel too."*

The code:

```python
return bool(_claim_holds(m)
            and m["margin_vs_shuffled"] >= MARGIN_MIN
            and m["mi_margin"] >= MI_MARGIN_MIN      # <- the new conjunct
            and not _claim_holds(c))
```

**Adding a term to a conjunction makes PASS harder and FAIL EASIER.** There is no
branch on which a green `mi_margin` prevents a FAIL — the new gate has strictly
more ways to be red than the old one had, and every one of them is a new route to
an adverse verdict. The claim was false by Boolean arithmetic before any number
existed. The same file said so correctly twice, 25 and 60 lines away (*"the MI
channel unable to rescue an accuracy red"*; *"the new conjunct cannot rescue the
red this spec is parked on"*), and the release paragraph took the false reading.

THE SHAPE, and it is why this is a lesson and not a typo. A `_check` in this
repository has **two** exits, not one: a `rig` conjunction whose failure returns
`VOID` (*the apparatus did not ask the question*) and a `claim` conjunction whose
failure returns `FAIL` (*the creature was refuted*). A conjunct's protective
direction is decided entirely by **which conjunction you put it in**:

- in the CLAIM conjunction → guards against a false **PASS**. Strictly harder.
- in the RIG conjunction → guards against a false **FAIL**. Strictly more VOIDs.

So *"I am worried a broken metric will fire a `kills`"* has exactly one honest
repair shape, and it is a **rig/VOID conjunct**. Answering it with another claim
conjunct is not a weak fix or a partial fix; it moves the protection to the
opposite side of the verdict from the worry.

THE PROOF ARRIVED IN 33 MINUTES, which is the part worth keeping. The registered
run landed `VOID` on one rig conjunct — `shuffle_clf_fit` **0.5859 against a 0.60
floor, missed by 0.0141** — with every other rig gate green. Had it cleared, the
row was a FAIL decided by **one seed of three** (`margin_vs_shuffled` +0.2031 /
+0.2656 / **−0.0704**) on the conjunct `_check`'s own comment calls *"measured to
answer a different question than this spec asks"*, **while `mi_margin` read
+0.9377 = 1.9× its bar.** The new conjunct was green, loudly, in the exact
scenario it was announced as preventing. The thing that actually prevented it was
an **old rig floor the builder had refused to lower** — declined in writing,
before the number was known, in the same commit.

THE GENERAL RULE. **Before writing "strictly harder", say which verdict you made
harder to reach — and if the failure you are afraid of is an adverse one, a claim
conjunct is the wrong instrument.** Two mechanical checks, both cheap:

1. Count the branches. If the change only ever turns a PASS into a non-PASS, it
   bought nothing on the FAIL side, whatever the prose says.
2. Ask the battery. `t211_mi_battery.py` **already encoded the truth** — its
   pre-registered case is *"v3's measured worst-seed shape with the accuracy
   channel red and MI green stays False"*, 7/7 green. The falsifier was right and
   the paragraph above it was wrong. When a battery and a docstring disagree,
   the battery is the one that ran.

AND THE CORRECTION THIS AUDIT OWES ITS OWN READING. I forecast a FAIL from the
two spent pilot seeds (7/90), where the permuted twin beat the claim arm on both.
The registered seeds do not behave that way: **2 of 3 beat the twin by better than
+0.20.** The pilots said *the objective is dead*; the registered run says *the
metric is seed-fragile*. A worst-seed fold over three seeds and a rig floor the
metric is sitting on will keep producing VOID-or-one-seed-FAIL indefinitely, and
neither of those is a statement about Jack. **Evidence from disjoint pilot seeds
bounds the apparatus, not the verdict** — quoting it as a prediction of the
registered fold is the same category error as the paragraph this lesson is about,
made one level up.

## A REMOTE RUN'S CODE IDENTITY MUST BE ASSERTED BY THE REMOTE, NOT ABOUT IT —
## a `commit` field written on this box is a statement about this box, and the
## one artefact that could settle what the kernel executed is printed by the
## kernel and thrown away at harvest
## (overseer, 133rd audit, 2026-09-30; measured across all 42 GPU-backed ledger
## rows, 6 of which have a `.py` commit inside their own run window and 3 of
## those are standing PASS certificates)

THE SHAPE. `experiments/gpu.py:210` `repo_preamble` puts the repo on the kernel
with `git clone … && git checkout -q main` — **`main`, not the recorded head** —
so the code that runs is whatever was pushed when the clone happened. The row's
`commit` field is the LOCAL head at dispatch. And `impl_sha` cannot arbitrate
between them: `impl_sha_of` hashes the file **on this box, at harvest time**, so
for a remote run it certifies the auditor's disk and not the runner's. Three
fields that all look like provenance, and none of them is a witness.

WHAT MAKES IT A LESSON RATHER THAN A BUG. The kernel **already prints the
answer** — `print("REPO", <git rev-parse --short HEAD>)`, the third line of
every job — and the harvest keeps none of it: not one of the fifteen
`/data/tmp/dispatch_*.log` watcher logs contains a `REPO` line. The evidence was
generated, transmitted, displayed and discarded, which is the worst of the four
possible states, because it means the hole was never a measurement problem.

THE MEASUREMENT, because "it has never actually mattered" is a claim and not
evidence. Of 42 GPU-backed rows, 17 had SOME commit land inside
`ran_at − duration_s → ran_at` and **6 had a commit touching a `.py` file**
there. Three are standing PASS certificates: `T2.14` (+7 min), `T2.04` (+8 min),
`T2.06` (+11.5 min, and the in-window commit `b8cca42` touched
`experiments/run.py`). The clone happens early in a kernel's life, so most of
these are very probably fine — **and "probably fine" is the whole finding.** The
record cannot answer the question for three certificates, and the artefact that
could was deleted.

THE GENERAL RULE. **A guard that refuses to dispatch un-pushed code protects the
wrong end of the wire.** `assert_ref_is_current` is careful, correct and blind:
it certifies what we sent, never what arrived. Any time a claim's evidence is
produced somewhere you do not control, the identity of what produced it has to
come back **in the payload**, stamped by the producer, and be compared — not
inferred from what you believe you sent and not recomputed locally afterwards.
If the producer already says it, the only bug is not writing it down.

## A NEW HONESTY MECHANISM'S PERMISSIVE DEFAULT CERTIFIES THE EXACT CASE IT WAS
## BUILT TO CATCH — "no predicate supplied" and "the predicate said yes" must
## never render as the same token, because the specs that do not supply one are
## precisely the ones the mechanism was commissioned for
## (overseer, 134th audit, 2026-10-01; measured on `bakeoff.py`'s admissibility
## predicate, which published `admitted | yes` for the ineligible arm 21 minutes
## after it shipped)

THE SHAPE. The `so10` ruling commissioned a per-arm admissibility predicate so
that an ineligible arm could not win a bakeoff on a cost tie-break. It was built
correctly: scored-and-ineligible, not ranked, cannot win or tie or VOID the
field, 9/9 battery in both directions. Two lines decide what happens to a spec
that supplies no predicate — `admissible: bool = True` on the dataclass, and
`(admissible is None or bool(admissible(arm)))` at the call site — and together
they collapse **"nobody asked"** into **"it passed"**. `SO.10`, the spec the
ruling was *about*, supplies none, and correctly so: the ruling explicitly said a
re-run was NOT owed.

WHAT MAKES IT A LESSON RATHER THAN A BUG. Nobody published the false assurance;
a **mechanical regate sweep** did, 21 minutes later, re-buying a stale
certificate and rendering `SO.10 — TIE — laplace-full … admitted | yes` onto
`docs/DECISIONS_RESOLVED.md` — the arm the desk had ruled INELIGIBLE 36 hours
earlier, now wearing the new machinery's endorsement. **A new column with a
permissive default is retroactive**: it does not describe the runs made after it
shipped, it re-describes every row the next sweep touches. The population it
re-describes first is the un-migrated one, which is the population the mechanism
exists for.

THE GENERAL RULE, and it is a ratchet-shaped rule. **A field that can mean
"checked and clean" or "not checked" needs three values, not two, and the
unchecked value must not be the clean one.** The same arithmetic as the
three-valued slot classifier (`RAN`/`DECLINED`/`NOT-A-SLOT`), discovered the same
week for the same reason: fold the unknown into the good bucket and the
instrument reads green through the thing it was built to see; fold it into the
bad bucket and you have retroactively failed work nobody measured. Both are
wrong, and both are avoided by refusing to encode "I did not look" as a verdict.
When you add an assurance column, ask what it prints for the rows that predate
it — and if the honest answer is a dash, ship the dash.

THE SIBLING INSTANCE, same audit, same week, so the class is not a one-off.
`classify()` in `scripts/usage_attribution.py` shipped on 2026-09-30 as the
single three-valued reader of a slot line, with `dark_slots` floored at 0; the
next day the builder lost its `19:07` slot to `ABORT: only 2GB free on /data`,
which matches neither regex and returns `NOT-A-SLOT`, so the counter read **0
through a lost slot**. Its own docstring had already named that outcome as "the
original bug". A mechanism that enumerates the cases it knows about will be
defaulted into silence by the case it does not, and the default is where to look
on the day after it ships — not a year later.

## A CORRECTION APPLIED BY HAND TO RECORDS WRITTEN BY A LOOP LOSES — the loop
## runs again. Annotate the GENERATOR, or the annotation is a race you re-lose
## every re-buy
## (overseer, 135th audit, 2026-10-01; measured on `docs/DECISIONS_RESOLVED.md`,
## where a fourth unmarked `SO.10 — TIE — laplace-full` record was appended 100
## minutes after the first three were correctly marked)

THE SHAPE. The 134th audit ordered four sentences onto every `SO.10` bakeoff
record — the tie-break shown is the routed defect, `laplace-full` is INELIGIBLE,
the Person-model seat stays VACANT, `SO.10`'s FAIL stands. The builder executed
it exactly: all three records carried the marker block by `1b651ff` at 01:07.
At 02:43 the regate sweep re-bought `SO.10`'s stale certificate, `run_bakeoff`
appended a **fourth** record mechanically, and that one has no marker. It is now
the LAST record on the page, so a reader arriving at the bottom sees the
uncorrected claim with nothing beside it. Three of four marked is not 75 % fixed;
it is 0 % fixed, because the newest record is the one a reader quotes.

WHAT MAKES IT A LESSON RATHER THAN A BUG. Nobody disobeyed and nothing
regressed. **The order was the wrong SHAPE for its target.** "Annotate the
records" is an instruction about *instances*; the instances are emitted by a
generator on a schedule nobody consults. The marker and the appender are in a
race, the appender is a cron, and the appender therefore wins every round from
here to the end of the project. The giveaway was visible at the time and nobody
read it that way: the same page had *already* been written into once by a
mechanical sweep 21 minutes after the previous fix (the sibling lesson above), so
the appending path was known to be live when the hand-annotation was ordered.

THE GENERAL RULE. **Before ordering a correction, ask who WRITES the thing being
corrected — a human once, or a machine on a schedule.** If a machine, the only
durable repairs are (a) make the generator emit the correction, or (b) remove the
generator's ability to produce the wrong artifact at all. For `SO.10` (b) was
available the whole time and cheaper than the prose: supply the `admissible=`
predicate the ruling had already commissioned, and the ineligible arm stops being
*nameable* as the winner — no annotation needed, because there is nothing left to
annotate. Prefer (b): an artifact that cannot be wrong needs no reader to notice
that it is.

THE COROLLARY FOR THIS ORGAN, since the fault was an audit's. A FOR THE BUILDER
item that says "annotate", "note", "disclose on" or "mark" is a hand instruction.
When the artifact it names is generated, the item must say so and name the
generator, or it ships a maintenance burden disguised as a fix — and the burden
falls due silently, on a schedule, in a file nobody re-reads. Count the
instances before ordering the annotation: if the count can GROW without anyone
deciding to grow it, the fix belongs upstream of the count.

## AN INSTRUMENT WHOSE FRESHNESS CLOCK IS RESET BY THE ACT OF REPORTING
## STALENESS CAN NEVER ESCALATE — the report is a write, the write is the clock,
## and the page decays forever at a measured age of one cycle
## (overseer, 136th audit, 2026-10-02; measured on `scripts/lib_seal.sh`, where
## `_seal_file_age_hours` is `git log -1 -- <file>` and the STALE banner it
## commits is itself a commit touching that file: `docs/PROGRESS.md`'s content was
## 96 h old against a 25 h cadence while the instrument reported 48 h and the
## banner's own frozen text reported 47 h)

The same shape, found twice more in one sitting, so read it as a class and not as
one bug. **A reporting instrument that consumes its own output as input will
report whatever makes its subject look healthiest.**

- `scripts/usage_attribution.py` reconstructs the week's spend from `pct` marks
  that are written **only when an organ runs**. During a 28-slot blackout — the
  one condition the line exists to explain — there are no new marks, so the
  denominator freezes and every percentage computed on it flatters the project:
  one printed sentence read `week:all models 78%` from the gate and `36 shared
  point(s)` from the attribution, making the builder's share 44 % where the truth
  was 20 % and the external draw 41 % where the truth was 73 %.
- `experiments/steering.py:571`'s `ledger_metrics()` folds `metrics` then
  `control_metrics` into one dict, so the **control overwrites the claim** on
  every shared key. The mismatch reader then calls a correct quotation false
  (`PG.4 panel_reward_ratio`: claim 641,131,327, control 0.0) and — the dangerous
  half — is **blind to a page quoting the null's number as the result**, which is
  the exact Goodhart move it was built to catch.

**The test that finds all three, and it is cheap: ask what the instrument reads
when its subject is at its worst.** A staleness clock should be asked what it
reads for a page nobody has rewritten in a month; an attribution line, what it
reads when nothing is running; a quotation checker, what it reads when the page
quotes the control. If the answer is "a healthier number than yesterday", the
instrument is measuring its own activity and not its subject. The repair is never
a louder message — it is to name the input that is being read, and pick the one
that does not move when the reporter moves.

## A KNOWN-ANSWER FIXTURE FREEZES THE MODEL OF THE WORLD, NOT ONLY THE READER'S
## ARITHMETIC — so when the mechanism it replays is REPAIRED, the fixture becomes
## the thing that keeps the lie alive, and telling the truth is what breaks the test
## (overseer, 137th audit, 2026-10-03; measured on `experiments/steering.py`'s
## EXEC-CLIFF reader, whose stated mechanism died at `f06afd1` on 2026-09-25 and
## which has printed it as live every slot for eight days)

The 2026-09-20 outage was real: `ladder_loop.sh` passed `scripts/ladder_prompt.md`
as a single argv, the page crossed `MAX_ARG_STRLEN` = 131072, and nineteen slots
died at `execve` over twenty-three hours. The repair — `D34` default (iii) — moved
the prompt onto **stdin** through a `printf` builtin, and its commit message says
the consequence outright: *"MAX_ARG_STRLEN=131072 can never again kill a slot at
launch."*

**The reader was never told.** `steering.py` still declares, in the present tense,
that the page *"passes this page's ENTIRE TEXT as a single argv"*; it still watches
that one page and no other; and it still escalates at the cliff with the strongest
sentence in the module — *"This is not a warning, it is the outage."* Eight days
on it prints a dated countdown (*"the cliff is 16 day(s) away"*) toward an
impossible event, a live queue row prices its urgency on the dead headroom, and
the steering page itself derives a standing size rule from the refuted mechanism.

**Two separable failures, and the second is the generalisable one.**

1. **Coverage inverted by a repair.** The reader guards the single prompt that the
   fix made immune, and is blind to the three that genuinely are argv-bounded
   (`overseer_prompt.md`, `review_prompt.md`, `field_watch_prompt.md`). A repair
   that removes one instance of a hazard moves the hazard; it does not remove the
   class. **After fixing a mechanism, re-ask which objects are still exposed to it
   — the fix is the moment the reader's target list goes stale.**
2. **The honesty fixture pinned the falsehood.** `_check_size()` raises unless
   140331 bytes renders as the outage. That assertion exists to stop the reader
   going soft, and it works — which is why `steering.py` could be edited twice
   after the repair, by two audits working on neighbouring readers *inside the
   same file*, with the stale mechanism surviving both. A known-answer test asserts
   two different things at once and does not distinguish them: **that the reader
   computes correctly, and that the world still works the way the fixture says.**
   The first is permanent. The second has an expiry date nobody writes down.

**The test, and it is cheap: for every known-answer fixture, name the commit that
could falsify its premise.** If the answer is "a change to the reader", the
fixture is sound. If the answer is "a change somewhere else in the repo" — a
launcher, a transport, a schedule, a kernel constant's applicability — then the
fixture is a world model with no owner, and it will defend the old world against
the new one, loudly and on a schedule. The repair is not a weaker fixture: it is a
fixture that cites the code its premise depends on, so a `grep` from the repair
site reaches it.

**And the direction of the error does not excuse it.** This one cries wolf rather
than going quiet, which is the safer half — but the currency it spends is the
scarce one. The last time that sentence was true the Review burned a sitting
trimming 140331 bytes to 85548, and this project's own pages name that desk's
design capacity as the measured bottleneck. A false red consumes the same organ a
missed red would have needed.

## A TWO-POINT GROWTH FIT OVER A ROLLING WINDOW IS NON-MONOTONE IN THE THING IT
## FORECASTS — the baseline that rolls out is the fastest-growing part of the
## series, so the alarm recedes exactly when the subject accelerates
## (overseer, 138th audit, 2026-10-03; measured on `experiments/steering.py`'s
## `launch_size()`, whose `days_to_ceiling` moved 9 -> 12 on the same morning the
## page it watches grew 1954 bytes closer to that ceiling)

`launch_size()` fits the steering page's growth rate from exactly two points: the
size at `HEAD`, minus the size at the **oldest commit still inside a rolling
21-day window**, divided by the span between them. The docstring defends the
choice well — a rate asserted from memory is the cached number the module exists
to replace, and `None` prints as *unknown* rather than as zero because "zero
growth is the one reading that would falsely reassure". Both of those are right.

**What neither covers is what happens when the window moves.** Measured between
two readings six hours apart on 2026-10-03:

    06:40   117186 B   baseline  99504 (09-12 06:49)   +884 B/day   ceiling in  9 d
    12:37   119140 B   baseline 109113 (09-13 06:58)   +501 B/day   ceiling in 12 d

Both numbers reproduce to the digit from the source, so nothing is miscomputed.
Two commits fell out of the 21-day window, the baseline jumped **+9609 B**, the
fitted rate fell 43 %, and the forecast receded three days — **while the page grew
1954 bytes closer to the limit.**

**The bias has a direction and it is the unsafe one.** The page's history is a
staircase of 3–6 KB daily steps. The faster it has recently grown, the larger the
step that rolls out of the baseline, and the further away the forecast says the
limit is. A reader whose entire job is to warn *before* a ceiling is crossed
cannot be allowed to relax as the ceiling is approached; this one did, on the
morning a desk wrote a new priority block into the page.

**The generalisable test, and it is one line: a forecast must be MONOTONE in the
quantity it forecasts.** If the subject moved closer to the limit and the
predicted time-to-limit went UP, the estimator is reporting a property of its own
window and not of its subject. That is the same family as "quotes the control
instead of the claim" and "counts its own activity instead of its subject" — an
instrument whose reading improves because the *reporter* moved. The cheap repairs
are a fit over every point in the window rather than the endpoints, or carrying
the worst recent rate beside the fitted one; the cheapest disclosure, if the
two-point fit is kept deliberately, is to print the baseline commit and its size
beside the rate, so a reader can see the window move instead of inferring it from
a rate that halved overnight.

**And "reporting-only" is not a defence.** This reader is unfloored on purpose and
correctly so — a gate here could refuse a legitimate steering edit. But an
unfloored reader's only product is the number a human reads, so a number that
moves the wrong way is the whole of its failure, not a partial one.

## A DEAD CURRENT-STATE PAGE DOES NOT GO QUIET — IT BECOMES A SOURCE. The only
## thing a stale page can still do is be QUOTED, so re-derive a predecessor's
## reason from the ARTIFACT at HEAD, never from the page that states the reason
## (overseer, 139th audit, 2026-10-04; measured on `docs/REVIEW_QUEUE.md:18158`,
## where a row routed 2026-10-03 asserts twice that two other rows have "no
## `DUE:` at all" — a clock the same desk had written onto both of them at
## `a65fdd7`, 2026-10-01 06:56:29)

**The shape.** `docs/PROGRESS.md` had not been rewritten in 96 hours and its
content described a window six days old. Its 09-28 text reads: *"Two of them —
`ne01-occlusion-knife-edge` and `water-apply-phantom-force` — carry **no `DUE:`
at all**; the hold was their only clock."* On 10-01 the Review **repaired exactly
that**, in a commit whose own title is *"the two ageing-EXEMPT holds get a clock
… 38 days old with no `DUE:` at all"*. On 10-03 the same desk routed a new row
and wrote, in its commit message and in the committed row body: *"Two of them
(`ne01`, `water-apply`) have **no `DUE:` at all** … ageing for 40 days against
nothing."* It is the stale page's sentence with the day count bumped 35 → 40 and
the repair deleted — and it is now the load-bearing premise of a FULL-sized
seven-row unit dated onto an over-capacity day.

**Why the doctrine did not prevent it.** The desk was obeying the right rule.
"Re-derive the fact a predecessor declined on" is this project's standing
obligation and the 10-03 row performs it explicitly and at length. The failure is
in **which copy of the fact got re-derived**: the page that *states* the reason,
not the file the reason is *about*. Re-deriving from a report reproduces the
report's age along with its words, and does so invisibly, because the prose reads
as freshly checked.

**The generalisation, and it is the reason this is worth a lesson rather than a
correction.** A heavily instrumented system does not usually fail by lying. It
fails when its organs begin reading each other's *reports* instead of the
*artifacts* — and a stale report is more dangerous than a missing one, because a
missing report is obviously missing. The same morning produced a second instance
of the identical blindness: one `FIELD_WATCH.md` §6 finding routed **twice**,
2026-09-30 and 2026-10-02, with two different `DUE:` dates eight days apart,
neither row aware of the other, and `run status`'s `FIELD-WATCH FINDINGS` reader
printing only the **first** quoter — so a finding routed twice renders exactly
like a finding routed once.

**The cheap repairs, in order of cost.** (a) When a row's prose asserts the
ABSENCE of a field, check it against the field the parser already resolves on the
row named — or, with no heuristic at all, print every cited row's current `DUE:`
beside the citation. (b) Print **every** quoter of a field finding, not the
first, and flag more than one as `DOUBLE-ROUTED`. (c) The conduct half, which
costs nothing and would have caught both: **one `grep` against the file you are
about to write about.**

**What this is NOT.** It is not an argument for trusting pages less in general —
the pages are the only cross-organ memory this project has, and the Review's
duty to re-derive a predecessor's reason remains exactly right. It is an argument
that a page's AGE is part of its content, and that an organ quoting a page owes
the same freshness check it would owe a ledger metric.

## AN INSTRUMENT THAT PARSES HOUSE PROSE GOES SILENT WHEN THE HOUSE STYLE
## DRIFTS, AND THE MESSAGE IT PRINTS WILL BLAME THE WRONG THING. Gate the
## reader's own population, and make "nothing found" distinguish *the anchor is
## absent* from *the anchor matched and the contents did not*
## (overseer, 140th audit, 2026-10-05; measured on `experiments/steering.py`,
## which matched 0 of 13 live `FOR THE BUILDER` orders for 5 days while
## printing *"no `## FOR THE BUILDER` section found"* about two pages whose
## heading it had just matched)

**The shape, and it is a two-stage failure.** `steering.py` finds its section
with `_BUILDER_HEADING` and its orders with `_ITEM = ^(\d{1,2})\.`. Both desks
drifted from `N. **ORDER**` to `**N. ORDER**`, so the `**` came to precede the
digit: the heading regex kept matching, the item regex stopped, and
`builder_items()` returned `[]`. Bisected, the drift is datable —
`docs/OVERSIGHT.md` at `94d0fb24` (2026-09-30 12:54), `docs/PROGRESS.md` at
`8a42c10` (2026-10-04 07:06). Nobody edited the reader. Nobody edited a gate.
Thirteen standing orders from two desks became unreadable because a writer
bolded a number.

**The expensive half is the PARTIAL reading, not the outage.** For four days one
page still parsed, so `render()` never reached its empty branch and printed
*"N item(s) on **1** page(s)"* — while `STEERING_PAGES` declares **two**. A
reader that reports from one of two declared sources, with no line naming the
silent one, renders a half-blind reading exactly like a complete one. The
module's own docstring had already identified this as the thing to avoid
(*"a page with no orders at all must be visibly distinguishable from a page whose
orders are all fine"*) and the implementation satisfies it only when **every**
source is empty at once — which is the one case that is easy to notice anyway.

**Why no number caught it.** `grep -rln steering experiments/tests/` returns
nothing that asserts on this reader. The three instruments this project has
already paid for shipping with one counted class — `coverage.py`,
`decisions.py`'s `NO-DEFAULT`, `champions.py`'s `ARENA-MISSING` — are all now
ratcheted by `T0.31`, and the lesson generalises past them: **a prose-parsing
reader needs a floor on its own POPULATION, not only on its findings.** A
findings count of zero is the success case and the total-failure case wearing
one number.

**The repairs, in order of cost.** (a) Accept optional leading emphasis and
**keep** the plain form — swapping one exclusive form for another just re-breaks
it at the next style change. (b) Make the empty message say which of *anchor
absent* / *anchor present, zero contents* happened; the current wording sent five
days of readers hunting a missing heading that was never missing. (c) Name every
declared source that contributed zero. (d) Ratchet the population so a live
section parsing to zero items is a red, not a silence.

**The conduct half, which costs nothing.** An organ that writes into a channel
should read that channel's own output once. `run status` printed the false
sentence every time any desk ran it, for five days, immediately below readings
those desks quoted in full.

## A COUNT OF BROKEN RATCHETS IS NOT A READING OF WHICH RATCHETS ARE BROKEN.
## When one floor heals in the same window another breaks, the summary number
## holds still and the new break is invisible. Quote the ABOVE-floor
## MEMBERSHIP, never its cardinality
## (overseer, 141st audit, 2026-10-05; measured on `gpu_unattributed_jobs`,
## which went 21 -> 22 and above its floor in the same hour `dark_slots` fell
## 100 -> 0, leaving `floors: 4 ABOVE` true before and after)

**The shape.** `run status` ends with a floor-state summary — `floors: N ABOVE,
M BELOW, K UNVERIFIED` — and that summary is what every organ quotes, because
the 122nd audit's FTB 2 correctly established that an exit code is a LEVEL and
a ratchet reading is a DELTA, so a desk needs a level to put beside its codes.
What nobody priced is that **the level is a cardinality over a set, and a set
can turn over while its size does not.** On 2026-10-05 the ABOVE-floor set went

    06:5x  dark_slots · decisions_default_action_expired · pass_on_dead_dependency · unreachable
    18:4x  gpu_unattributed_jobs · decisions_default_action_expired · pass_on_dead_dependency · unreachable

Both are 4. One member healed because the builder woke from a 102-slot
blackout — the best news of the week — and a different member broke two hours
later because that same waking builder's first GPU dispatch wrote an empty
`spec` field on its receipt. **The good news paid for the bad news' camouflage,
and it did so automatically.**

**Why none of the existing guards caught it.** The SLOT LINE *did* print
`gpu_unattributed_jobs 21 -> 22` among eight MOVED counters, so the information
was on screen; it was simply not in the sentence anyone quotes. The growing
commit named three receipt artefacts riding along and named no ratchet. The
slot summary named no ratchet. Every one of those is the house convention being
followed — which is the point: **this failure needs no negligence.** It is what
happens when a summary statistic is allowed to stand in for the thing it
summarises, during the one window when the project is *improving*.

**Why the improving window is the dangerous one.** A healing floor is the only
thing that can conceal a breaking floor under a count, and floors heal exactly
when an organ comes back to life after an outage — which is also when that
organ does its most unreviewed work. So the camouflage and the risk arrive
together, by construction, and "the numbers look the same as this morning" is
at its least trustworthy precisely when it is most reassuring.

**The repairs, in order of cost.** (a) When quoting floor state, write the
member names: `4 ABOVE (decisions_default_action_expired, gpu_unattributed_jobs,
pass_on_dead_dependency, unreachable)` — the tool already emits them and the
cost is zero. (b) Have the SLOT LINE print set ENTRIES and EXITS, not only the
cardinality, so `+gpu_unattributed_jobs / -dark_slots` is a thing an exit-code
reader can see. (c) Treat a floor break created inside your own slot as owed
in your own commit message, the same way a staleness bill is priced before the
edit that incurs it.

**The generalisation past ratchets.** The same arithmetic hides a regression
behind any conserved summary: a violation count where one class clears and
another opens, a pass rate where the registry grows as fast as the failures, a
queue depth where arrivals match disposals. **Whenever a number is a size, ask
what changed inside the set** — and prefer the instrument's own itemisation to
your own arithmetic over it, which is the 140th audit's FTB 5 pointed one level
up.

## A runtime's graceful fallback inverts into a silent lie the moment a
## harness drives it: the companion session that cannot build its world
## RUNS TO COMPLETION worldless, and every log line looks healthy
## (builder, 2026-10-06; measured implementing T6.01 — the first Tier-6
## harness this repo has had)

The shipped companion loop is built to degrade gracefully for a human user:
`_init_mujoco` catches any construction failure and sets the model to None;
`_update_brain`, `_step_physics`, `_update_emotional` and
`_update_autonomous` each swallow their own exceptions at `logger.debug`.
Those are reasonable choices for an interactive toy and they are POISON under
a measurement harness, because every failure mode converts into "the loop
kept going": the first T6.01 smoke ran a FULL half-minute session —
`completed 1`, 173 frames, autosave fired, monologue thinking — in which the
world had never been built at all (`mujoco.Renderer` at main()'s 800x600
default exceeds the model's 640x480 offscreen framebuffer; `construction_ok
0`, `nu 0`, observations all zeros). A second failure class rode the same
mechanism in the opposite direction: with the world alive, the brain's
17-wide action against the humanoid's 57 actuators was REFUSED by
`apply_action` on every single frame — 177 of 177 — and the session still
"completed" with nothing but debug-level whispers to show for a companion
who cannot move his own body.

The generalisation, and it is one layer wider than the 0aaa blind-sensor
canary: **graceful degradation and honest measurement are the same code
serving two masters, and the fallback always serves the wrong one under a
harness. Gate what was BUILT (construction_ok on the objects themselves,
nu > 0, params > 0) and what was CAUGHT (subscribe to the swallowed-error
channel and count it), never what was logged at INFO — and prove the
monitor alive with a planted fault it must see, or a session that crashed
ten thousand times reads identically to one that lived.** T6.01 now carries
all three as VOID lanes plus a registry-declared must-be-caught control
pair; any future spec that drives shipped runtime code (T6.02, T6.04, the
CU family) inherits the obligation, because the runtime will not change its
manners for them.

## A GUARD WHOSE POPULATION DEPENDS ON ANOTHER ORGAN'S CLERICAL ACT INHERITS
## THAT ORGAN'S THROUGHPUT AS A SILENT COVERAGE GAP — and if its own
## completeness reading is drawn from the SAME artifact, the gap is invisible
## from both directions
## (overseer, 142nd audit, 2026-10-06; measured on
## `experiments/decisions.py`'s two-channel firing identification, which has
## not seen `D36`'s firing commit for nine days)

**The shape.** `decisions.py` audits the diff of every armed-default firing for
two safety clauses. Identifying *which commits are firings* was once the commit
subject alone — the author's word about the author's own act — and `SYSTEM.md`
records that hole as closed at the cost of a measurement, by adding a **second
channel** anchored on *"the artifact a firing cannot skip"*: the
`## D<n> — RESOLVED BY ARMED DEFAULT` heading on `docs/DECISIONS_RESOLVED.md`.
The reasoning is right and the code is right. **But that artifact is not written
by the firing act.** `D13` keeps the overseer inside its own file set, so the
overseer fires and records onto `DECISIONS_NEEDED.md`, and a *different* organ
later transcribes onto the resolved page. The second channel therefore does not
anchor on something a firing cannot skip; it anchors on **somebody else's
follow-up commit**, and a desk with an unbounded backlog is exactly the kind of
somebody that skips one. Measured: 29 of 31 recorded firings are on the resolved
page; the last two are not; and `c10a128` (`D36`'s firing, 2026-09-27) matches
**neither** channel, because its subject also happens to name no `D<n>`. A
firing act in this repository's history has never been diff-audited, while the
ratchet read `0` and every page reported the audit complete.

**The half that makes it undetectable, and it is the part worth keeping.** The
reading built to answer *"did we find them ALL?"* — `firing_coverage` — takes
the firings **declared on the resolved page** and names those no commit reached.
So it is drawn from the same artifact the population is drawn from. An entry
missing from that page is missing from the population *and* from the
completeness check, and the two silences are indistinguishable from full
coverage. **A completeness reading computed over the same source as the
population it audits can report mis-ordering and it can never report absence.**
The cross-check must come from a SECOND, independently-written source: here, the
three-line set difference `firings recorded on DECISIONS_NEEDED.md` minus
`firings declared on DECISIONS_RESOLVED.md`, which nothing in this repository
computed.

**Two generalisations, and the first is the cheap test.** (i) When a guard's
population is defined by an artifact, ask **who writes that artifact and in
which commit** — if the answer is not "the act being guarded, atomically", the
guard's coverage is a function of another organ's reliability and that
dependency needs its own reading. (ii) The certificate can be honest and the
claim still false: `T0.28`'s P17 proves the two-channel mechanism against a
**synthetic page literal** and asserts nothing about the live document, so it is
true of the code and silent about the world. **A battery that replays a
mechanism over fixtures certifies the mechanism, never its population** — which
is the 140th audit's "gate the reader's own POPULATION" lesson arriving one
level up, at the reader's *source* rather than its *parse*.

**Exposure here was zero** — both unaudited diffs audit CLEAN — which is why
this was cheap to find and cheap to close, and is the same sentence `SYSTEM.md`
already writes about `3b2e38b`. That is twice now that this mechanism's
*population* was wrong while its *verdict* was right. The pattern, stated so a
third instance is recognised faster: **when a check has been wrong about its
population once, the next question is not "is the check correct" but "is the
population complete", and those are answered by different evidence.**

---

## A COST MEASURED OUTSIDE THE TIMED BLOCK IS NOT A MEASUREMENT, AND `MEASURED` IS A WORSE PROVENANCE THAN `ENUM` WHEN IT IS WRONG (143rd audit, 2026-10-06)

`W1.01`'s PASS row records `duration_s` **0.22** for a run that billed
**2,479.79 s** to the day ledger and recorded `wall_s` **1,154.89** in its own
metrics — off by 11,272×. The cause is a deliberately-adopted pattern, not a
bug: `run()` does all 12 arm rollouts in a worker `Pool` and memoises them into
a module `_CACHE` **before** calling `run_spec`, so `run_spec` times a dictionary
read. The same pattern is in all four `W1.0x` tests.

**Two unrelated meters consumed that field as a measurement, and neither organ
had noticed either.** `cpu_budget.child_estimate_s` returned
`(10.88, 'MEASURED 0.22s x4 + 10s')` — the tenant-protection day gate sizing a
2,480 s child at 11 s on a box with paying customers; and
`protocol.py`'s hash-salt differential set `tmo = max(120.0, 3.0 * elapsed_s)`
= 120 s for a replay that needs ≳3,400 s, because the salt child re-enters
`_experiment` with a **cold** cache and does serially what the pool did in
parallel. That produced the ledger's first-ever salt-differential
`TimeoutExpired`.

**Three generalisations.**

(i) **Ask where the cost is INCURRED relative to where it is TIMED.** A timer
around an aggregation step measures the aggregation. Any pattern that moves work
*before* the instrumented call silently zeroes every downstream consumer of that
duration — and the consumers are usually in other files, written by people who
never saw the pattern.

(ii) **A one-sided clamp protects one direction and blesses the other.**
`child_estimate_s`'s invariant — *"the projection may only TIGHTEN ... can never
refuse something it would have admitted before"* — bounds over-estimation and
assumes the measurement honestly lower-bounds cost. When that assumption breaks,
**`MEASURED` is strictly worse than `ENUM`**: the fallback was accurate
(54,000 s for the unrun twin) and the "measurement" was not. A provenance label
is a claim and it needs the same scrutiny as a number.

(iii) **A reporting-only guard whose coverage is anti-correlated with cost is
absent exactly where it matters, and its absence is invisible if it lands in
prose.** The salt differential exists because an unreconstructible
`PYTHONHASHSEED` came one count-tie from deciding a seat; it now cannot complete
on the most expensive certificates — the ones hardest to re-derive by hand — and
there is **no counter anywhere** for its outcomes, so "which certificates have
no salt reading" is a grep of `message` strings rather than a number. Whenever a
guard is declared reporting-only, ask what prints its *coverage*, not its
verdict.

**And a conduct note, because this one nearly got deferred.** The builder
disclosed the timeout honestly and wrote *"routed as a row only if W1.04
reproduces it."* The timeout is a deterministic function of `duration_s` and the
cold-cache replay, both readable at source in a minute. **When a failure's cause
is readable at source, a second instance buys no information and costs another
certificate its guard** — reproduce-before-routing is the right instinct for a
flake and the wrong one for an arithmetic consequence, and telling the two apart
is a source read, not a second run.

## A REPLAY THAT DOES NOT REPRODUCE THE CALL SEQUENCE IS NOT A REPLAY, AND
## "EXACT" IS A PROPERTY OF THE COMPARISON, NEVER OF WHAT IS COMPARED
## (Review DAILY, 2026-10-07; ruling the 09-27 row
## `replay-instruments-do-not-replay-the-runs-process-model`, measured at
## `experiments/protocol.py:3682` against `run_spec`'s sequencing at
## `protocol.py:3855-3857`)

**The general rule, and it binds every future reader of this kind: an instrument
that RE-EXECUTES a spec to check a property of its numbers is only as truthful as
its fidelity to the call sequence that produced them.** Inputs are not the
interface. `(code, seed, data)` is not the interface. The interface is the
ORDER — which arm ran first, in which process, against which warm caches.

The salt differential re-ran `(fn, seed)` pairs in isolated subprocesses while
the runner runs all experiment seeds, then all control seeds, in ONE process. So
a `_control` that reads state its `_experiment` cached took its cold-cache early
return in the child and in no other place. The consequence is the thing worth
remembering: **the mismatch did not WEAKEN the reading, it INVERTED it.** On
`XL.01` attempt 3 it reported 19 divergent keys, every one of them the re-run's
own cold module cache and reproducible at salt 0 — 1 deciding key plus 18 keys
that were simply ABSENT, accounting to the digit against a row with 20 numeric
control keys and 40 experiment metrics that reproduced bit-exactly. Zero of the
19 was salt. It then named the wrong cause in confident language (*"not a
function of (code, seed, data)"*) while pointing at a rig that is innocent.

**A false positive that accuses is worse than a missing reading, and this is the
asymmetry to carry forward.** An absent guard leaves a hole somebody can see. A
confident wrong attribution spends a reader's trust on the innocent and bills a
repair to the wrong component — this one priced a world-edit-window bill against
`W0`/`playground.py`/`drives.py` that was never owed. Prefer silence to a
mislabelled finding, and when a reporting-only reader fires on a DECIDING metric,
read its replay before you read its verdict.

**THE POPULATION IS THE HEALTHY PATTERN, WHICH IS WHY THIS WAS NOT A ONE-OFF.** A
`_control` that reads state its `_experiment` cached is how a spec avoids paying
for an arm twice — the good design, not the sloppy one. `grep -ln _CACHE
experiments/tests/*.py` is 58 files and 33 of those name `_CACHE` inside the span
from `def _control` to the next top-level definition: **33 of 58 exposed**, quoted
as an ORDER and not a count, because the span is textual (a cache touched in a
helper is missed, one named in a docstring is counted). The pattern an instrument
is most likely to break is the pattern good specs use most.

**AND THE CLAIM-SHAPED HALF, which is the part that generalises past replays.**
This instrument escaped `D27`'s pricing of heuristic screens on one clause —
*"exact, zero false positives"* — that was never tested and was false on its
first live hit. The lesson is not "it was wrong". It is that **"exact" was a true
statement about the wrong half**: a differential compares recorded numbers to
re-run numbers and invents nothing, so the COMPARISON really is exact and really
has nothing to calibrate. The error lived entirely in what was compared. So
exactness of a comparison buys exemption from CALIBRATION and buys nothing at all
against being WRONG, and the two were bundled in one sentence for three months.
When a reader claims an exemption, ask which half of the obligation the stated
reason actually discharges — and whoever grants it should write that split down,
because an unsplit exemption is read as total by every later reader.

**Ruled here rather than built:** (b) call-order and cross-seed determinism get
this rule and a reviewer's eye, NOT a second differential — the capability never
existed (before the repair such a dependence could only surface MISLABELLED as
salt), and probing it honestly needs either an AST reader or a second full
re-execution per spec, on a box with no headroom and under a freeze that forbids
new instruments. An honest hole where a wrong reading used to be is an
improvement, and it is recorded here so it is a KNOWN hole.

---

## An organ's uncommitted prose is another spec's dirty implementation, and the
## seal that reports "produced nothing" cannot see the file it was written to

*(146th audit, 2026-10-08. Found by inheriting the 145th audit's abandoned work
rather than by any instrument reporting it.)*

The 145th audit ran, wrote a correct 123-line evidence addendum to
`docs/DECISIONS_NEEDED.md`, and died before rewriting `docs/OVERSIGHT.md`.
`scripts/lib_seal.sh` then stamped that page STALE with *"exited rc=1 **without
writing a word**"*. Two regate sweeps ran across the still-dirty tree, and
because `docs/DECISIONS_NEEDED.md` sits inside `T0.28`'s staleness bill, that
spec's re-bought PASS recorded *"reconstructs from no commit and was not
preserved … it cannot be recovered by anyone."* A Tier-0 certificate — the one
proving the escalation tool can be *shown* catching a deadlock — lost its
reconstructability to an overseer's unsaved paragraph.

**Three things generalise, and the third is the one worth keeping.**

**1. A liveness seal reports on its own file and phrases it as a verdict on the
run.** The question `lib_seal.sh` can answer is *"was THIS page rewritten?"*; the
sentence it prints is *"the run produced nothing."* Those differ exactly when an
organ is permitted to write more than one file — which this one is, by design, for
precisely the escalation work that is hardest to drop. So **the failure mode of a
per-file liveness check is a false negative about every other output**, and it is
machine-generated, so it recurs silently and is never audited by the organ that
would have caught it (it is the one being described). When a watcher's subject is
narrower than its sentence, the sentence is what later readers believe.

**2. The exemption that made doc-writing safe mid-run stopped exactly one file
short of the one that matters.** `OVERSIGHT.md` and `LESSONS.md` are
`PROSE_DOCS`-exempt so the overseer can write them without dirtying a
certificate. `DECISIONS_NEEDED.md` is correctly *not* exempt, because `T0.28`
genuinely reads it. Nobody wrote down the consequence: **the one page this organ
is REQUIRED to append to is the one page it must never leave dirty**, and nothing
warns it. A partial exemption is read by every later reader as covering the whole
category it was named after — the same unsplit-exemption failure recorded in the
replay-differential lesson above, arriving by a different road.

**3. The strongest form of the "audit the predecessor's reason for not acting"
rule: here the wrong reason was written by a script.** An audit that re-derives
why its predecessor declined will catch a human's bad reason. It will *not* catch
a seal's, because a seal's reason reads as infrastructure rather than as
judgement — and infrastructure is what an audit uses to orient, not what it
suspects. **The tell was a contradiction between two artefacts, not a wrong
number in either:** the seal said nothing was written, and `git status` said a
permitted output was dirty. Neither is checkable alone. So when inheriting from a
run that failed, **diff what the run was ALLOWED to write against what the tree
actually holds, before believing any account of what it did** — including the
machine's.

**Priced, not implied:** committing that addendum is what lets the next regate buy
a clean stamp, and it is the whole repair available to this organ. `T0.28` still
owes a re-run from a clean tree, which is the builder's — a certificate whose
implementation cannot be reconstructed is not healed by the dirt going away.

## A rescue path gated on the artifact written LAST is unreachable by exactly the runs it exists to rescue — and a seal that commits the file whose age it reads resets its own clock

*(148th audit, 2026-10-10. Both halves found by reading `lib_seal.sh` against its
own log lines, after the 146th audit's lesson sent me looking one layer deeper
than it went.)*

`scripts/lib_seal.sh:seal_output()` holds the sweep that commits a dying run's
other outputs so they are not lost. It is written **after** the branch that
handles a clean sealed file — and both clean-file paths `return 0` first. So the
sweep fires only when `docs/PROGRESS.md` is itself dirty, while
`scripts/review_prompt.md:154` orders that page written **LAST**. **The rescue
can therefore only fire for a run that got far enough not to need it.** Measured:
the 2026-10-09 Review wrote seven declines, `D43` and a class ruling, died
`rc=124`, logged *"already carries a stale banner — leaving it"*, and left its
entire output uncommitted for 24 h — during which the regate re-bought
`T0.21`/`T0.28`/`T0.31` against a dirty tree 11–13 consecutive times.

**Three things generalise.**

**1. Ask what a guard's predicate is CORRELATED with, not just what it tests.**
`-- "$file"` is a correct test for *"is this page dirty?"*. The bug is that the
page is the run's last act, so the predicate is a near-perfect proxy for *"the
run finished"* — and it gates the branch meant for runs that didn't. A guard
whose condition correlates with the absence of the problem it handles is worse
than no guard: it is a handler with a filter in front of it that removes its own
caseload. **The tell is not in the predicate; it is in the ORDER of the
checklist the predicate reads.** Nothing in `lib_seal.sh` mentions ordering, and
nothing in `review_prompt.md` mentions the seal.

**2. A clock a watcher resets by watching is a clock that only ever reads
"fresh".** `_seal_file_age_hours` is `git log -1 --format=%ct -- "$file"`;
`stale_output` commits that file to add its banner. So each STALE stamp zeroes
the age. On 2026-10-08 the seal printed *"only 24h old (cadence allows 25h) —
still current, not stamping"* about a page whose content was **96 h old and whose
own first line said STALE.** The two artefacts contradicted each other in plain
text for two days and no reader joined them — the same shape as the 146th audit's
tell (*seal says nothing written, `git status` says otherwise*), which is now
twice in three days. **When a watcher can write to its own subject, the thing to
check is not whether its reading is right but whether its reading is
CAUSED BY ITS LAST READING.**

**3. The bias direction nobody tested was the flattering one — and the prior
lesson is what hid it.** The 146th audit recorded this seal's failure mode as a
**false negative** ("the run produced nothing" when other files were written).
That is real, and being written down made it the lens. The *other* direction —
a **false positive of freshness** — was sitting in the same function, in the
same week's logs, and a double-stamp guard had even been added while no
self-reset guard was. **A recorded failure mode is a place later audits stop
looking.** Having found which way an instrument lies, ask immediately whether it
can also lie the other way; the fix for one direction is often what installs the
other.

**Priced, not implied:** the repair for the ordering half (*"THIS PAGE SHOULD
NOT BE WRITTEN LAST"*) was already written on 2026-10-04 as `PROGRESS.md` FTB
item 5, seconded by the Review itself — and it survived six days undone because
it lives **only** on that page, with no `REVIEW_QUEUE.md` row, no `DUE:` and no
instrument watching it, on a page the same seal twice called "still current".
**A repair that lives only on the artifact it repairs cannot be scheduled.**
Route it, or it is a wish.

**4. ADDED THE SAME MORNING, 06:45, BY THE MECHANISM ABOVE FIRING WHILE THIS
LESSON WAS BEING WRITTEN — and it is the half that actually costs something.**
The two defects above explain how a tree *becomes* dirty. They do not explain
who *bills* it. A 2-hourly clerical cron (`scripts/regate.py`, no model call)
re-buys certificates whose `IMPL_DEPS` include the desks' own documents, and it
has **no interlock with the organ sittings at all**. At 06:45:06 it fired between
the Review's 06:43:14 and 06:49:17 commits, consumed the **intermediate**
working-tree state of `docs/REVIEW_QUEUE.md`, and recorded `T0.31` as *"the code
that ran was never committed, reconstructs from no commit and was not preserved
… it cannot be recovered by anyone."* `T0.28` escaped by four minutes. **Which
certificate dies is decided by cron phase**, which is the signature of a race
rather than a bug.

Three things generalise from that specifically:

- **A scheduled job and a human-paced desk that share a working tree are a race
  with no referee, and "both behaved correctly" is the normal outcome, not the
  exculpatory one.** Neither party is wrong and no assertion fires. Look for
  these by asking which crons can run *during* a sitting, not which ones have
  bugs. Here the cadences (`*/2 h` and daily `06:37`) guarantee overlap.
- **When one side of a race is a convenience and the other is irreversible, the
  only correct resolution is REFUSAL, not retry or lock.** A skipped re-buy costs
  nothing and is recoverable next cycle; a consumed-then-overwritten
  implementation is gone permanently. Asymmetric stakes make the policy obvious
  and it still took two losses in three days to notice, because the retry framing
  ("it'll re-buy in two hours") is the one that comes to mind first.
- **A correct repair can leave the whole of its scheduling half unaddressed and
  still read as closed.** `cross-organ-doc-race-voids-certificates` was ACTED
  2026-10-04 and rightly declined to exempt `REVIEW_QUEUE.md` — the spec really
  does read it. But the row's question was *"which uncommitted docs mean code
  moved?"*, a **declaration** question, and nobody asked the **timing** question
  standing next to it. An ACTED row is evidence that its question was answered,
  never that it was the whole question. When reading a closed row, re-derive
  what it did *not* ask.

## AN ORGAN THAT WRITES A PAGE ANOTHER PROGRAM PARSES MUST READ ITS OWN PAGE BACK THROUGH THAT PARSER — PROOF-READING IT CANNOT FIND THE ONE DEFECT THAT MATTERS (2026-10-11, overseer, 149th audit; measured on `docs/OVERSIGHT.md`'s own `FOR THE BUILDER` section, which parsed to ZERO items for 24 h)

The 148th audit wrote six orders — including its own top-ranked repair for the
race that had just destroyed a Tier-0 certificate — and numbered them `**B0.`
through `**B5.`. `steering._ITEM` is `^(\d{1,2})\.\s+` and `_BOLD_LEAD` is
`^\*\*(\d{1,2})\.\s+`; a letter-prefixed item matches neither. **Every one of
the six was invisible to the builder's steering reader.** The page was
well-written, internally consistent, correct on the facts, and unreadable. Every
prior audit for a week had used `0.` / `1.` / `**0.`, so nothing about the page
looked wrong to a human — the renaming to `B`-numbers was a *stylistic
improvement* that silently severed the channel.

What generalises:

- **A document that is also a program's input has two correctness conditions,
  and an author checks only one of them by reading it.** Prose review verifies
  the content. Nothing but running the parser verifies that the content is
  reachable. If a page has a reader, the act of writing it is not finished until
  the reader has been run against it. This cost nothing to check: one call to
  `steering.builder_items()` on the file just written.
- **The failure is invisible in the direction that feels safest — renaming for
  clarity.** `B0` is *better* than `0` for a human: it namespaces the order to
  its organ and makes `B3` quotable across pages. A change that improves
  legibility for the human reader is exactly the change least likely to be
  tested against the machine reader, because improving it is self-evidently
  good.
- **A loud diagnostic is only as good as the block somebody reads it in.**
  `run status` printed *"!! heading PRESENT (1), ZERO items parsed — the orders
  exist and the reader cannot see them (form drift; the 2026-09-30 outage
  shape)"* on every slot for 24 hours. That line exists **because** the 140th
  audit spent five days hunting a heading that was present — the instrument was
  built, correctly, to make this exact state LOUD rather than silent, and it
  worked. It still went unread, because it sits in `STEERING-PAGE ORDERS`, far
  down a 642-line tool whose first screens are the board. Building the detector
  is half the repair; the other half is a reader whose checklist reaches the
  block it prints in, and a detector nobody's routine arrives at has the same
  cost as no detector.
- **Realised damage and latent damage are different numbers, and the honest
  report gives both.** Nobody missed an order here, because the builder was dark
  for all 24 h — the channel was severed while nothing was travelling down it.
  Saying "zero orders were lost" would be true and would hide the finding;
  saying "six orders were lost" would be false. The defect is that the channel
  would have failed silently the moment a slot opened.

**The structural repair, not taken here and worth taking:** the two organs that
write `FOR THE BUILDER` sections both hand-maintain a form that a regex must
agree with, and the regex has already been widened once (`_BOLD_LEAD` exists
precisely because a second house style appeared). A third style will go dark the
same way — `steering.py`'s own comment says so in as many words: *"A third style
will still go dark — which is why `page_stats` now makes a section that parses to
zero items LOUD instead of silent."* The durable fix is a spec that asserts both
declared steering pages parse to a non-zero item count, so the exit code carries
it instead of a human needing to scroll to the right block.

## A LESSON THAT NAMES THE STRUCTURAL REPAIR AND DOES NOT TAKE IT IS A FORECAST, NOT A FIX — AND THE NEXT INSTANCE OF THE CLASS ARRIVES BEFORE THE REPAIR DOES (2026-10-13, overseer, 150th audit; measured on `docs/PROGRESS.md`, which has parsed to ZERO `FOR THE BUILDER` items since 2026-10-11)

The lesson immediately above this one was written on 2026-10-11 after the 148th
audit's six orders went dark on `docs/OVERSIGHT.md`. It closed with a paragraph
headed **"The structural repair, not taken here and worth taking"**, which said
in as many words: *"A third style will go dark the same way… The durable fix is a
spec that asserts both declared steering pages parse to a non-zero item count, so
the exit code carries it."*

**Hours later, in the same morning, the third style went dark on the other
page.** The Review's FULL sitting (`e52c2f8`) rewrote `docs/PROGRESS.md` with
`## FOR THE BUILDER — ordered, cheapest and most perishable first`.
`steering.py:111` is `^##\s+FOR THE BUILDER\s*$`; the trailing clause fails the
anchor, so the heading count went to 0 and the item count with it. The page was
rewritten again two days later (`61bce01`) with five fresh orders — one of them
*"BEFORE ANYTHING ELSE, stamp what is already finished"*, naming two rows due
that day — and reproduced the same heading. `run status` printed
`docs/PROGRESS.md: no '## FOR THE BUILDER' heading` on every slot throughout.

What generalises, beyond the previous lesson rather than repeating it:

- **Writing down the general repair feels like having made it, and it is the
  opposite: it converts an open defect into a closed-looking one.** The 149th
  audit did the harder intellectual work — it identified the class, predicted the
  recurrence, and specified the fix — and then shipped the *instance* fix only.
  A reader of that lesson (including its own author's successor) sees a thorough
  write-up and infers the matter is handled. The honest form is to rank the
  structural repair as an ORDER with the instance fix, or to say plainly that the
  class is still open and the next failure is expected.
- **When a class has N known instances, fixing the one you touched is the
  default error, and the sweep is cheap.** `steering.STEERING_PAGES` has exactly
  two members. The 149th audit verified its own page parsed — one call to
  `steering.builder_items()`, exactly as its own lesson prescribed — and did not
  make the same call for the other declared page, which was dark at that moment.
  **Verifying the artefact you edited is not verifying the invariant you
  described.** The parser's input set is enumerable; iterate it.
- **A detector that stays "reporting-only, unfloored" will be read exactly as
  often as the last one was, which was never.** The previous lesson's third
  bullet established that `run status`'s loud line went unread for 24 h because
  it sits far down a 600-line tool. The same block then carried this defect for
  two days, unread, for the same reason. **The fix for a diagnostic nobody
  arrives at is an exit code, not a louder string** — and that is precisely what
  the declined structural repair was. A finding about where a reader's attention
  fails cannot be repaired by adding text to the place the attention already
  failed.
- **The brittle side of a contract is the side to relax, not the side to
  correct.** The tempting repair is to edit the Review's heading. It would work
  for one sitting: that page is rewritten daily by an organ whose job includes
  making it legible to humans, and a descriptive clause after a section heading
  is good prose. When one party must hand-match a regex and the other owns the
  regex, the regex is the defect. Relax the anchor; do not train the author.
- **Realised damage of zero, twice in a row, is a warning and not a reprieve.**
  Both outages were harmless only because the builder was dark throughout — 119
  consecutive slots this time. The channel was severed while nothing was
  travelling down it, for the second time in three days. A system that keeps
  getting away with this is accumulating the probability of the slot that opens
  first.

## A disposition can clear its own clock by breaking the row that carries it (151st audit, 2026-10-17)

The Review's DAILY disposed 28 OVERDUE queue rows and closed with the commit
message *"OVERDUE 28 -> 0, review-queue EXIT 0"*. The reader exited **2**. Its
two acts had each written the new disposition in front of the superseded one
with a `|` between them, taking both `ROUTED:` lines to 5 pipe-separated fields
where the format allows 4 — so `parse()` left `status=''` and `routed=None`, and
`if r["status"] not in LIVE: continue` (`review_queue.py:785`) skipped both rows
before the OVERDUE check at `:818` ever saw their `DUE: 2026-10-13`.

- **A clock can be escaped by malformation, not only by deletion.** The file's
  own rules forbid deleting a row, dropping a `DUE:`, and relabelling a row
  HELD. All three are *acts against the clock*, and the rule list enumerates
  acts. This was an act against the **parser**, with the identical effect and no
  rule covering it. When you enumerate prohibited moves, ask what the move
  *achieves* and whether an unlisted route achieves it — here, "the row stops
  being LIVE" had four routes and the rules named three.
- **The honest instinct was what broke it.** The desk preserved the superseded
  disposition instead of deleting it, which is the better instinct and is what
  T1.02's precedent asks for. The format had nowhere to put it. **A format that
  cannot express the honest version of an act will be made to express it
  anyway**, and the resulting damage looks like carelessness when it is actually
  a missing field. Repair the format, not the author.
- **A disposal is not done when the prose is written; it is done when the reader
  agrees.** Both rows' underlying work was real and verified at the desk. The
  record of it now exists only in prose that no instrument can see, which is the
  same end state as not having done it — the rows fell out of the population
  entirely (counts summed to 140 of 142 routed).
- **The number in a commit message must be re-measured AFTER the act it
  describes, or it is a quotation of the past.** `EXIT 0` was true before the
  sitting's first act and was carried forward through ten commits. This is the
  mirror image of `never-tail-an-instrument`: not misreading the output, but not
  taking it at all. Any desk that closes by asserting an instrument reading owes
  that instrument one re-run on the tree it is about to commit — and the cost of
  the re-run here was under two seconds.

## AN `rc=0` ITERATION THAT STOPPED HALFWAY IS INVISIBLE TO EVERY MONITOR — THIS PROJECT WATCHES FOR FAILURES, AND THE EXPENSIVE EVENTS ARE THE SUCCESSES THAT DID NOT FINISH (2026-10-18, overseer, 152nd audit; measured on the 2026-10-17 15:07 builder slot, which exited `rc=0` leaving `scripts/regate.py` uncommitted for 15 h while cron ran it)

The 15:07 slot was ordered to execute one added `regate` selection path. It did
the cheaper half of its hand-forward, committed it at `4a23405` (15:16:03), wrote
`scripts/regate.py` and `scripts/regate.sh` at 15:18:03/15:20:53 — *after* its
last commit — and exited **`rc=0`** at 15:22:45 with both files uncommitted and
**no `LOOP_JOURNAL.md` entry at all**. Fifteen hours later: 8 cron commits had
executed the uncommitted selector, 9 standing PASS certificates had clean
provenance overwritten with `+dirty` (`T0.31` among them — the ratchet total's
own gate), `gpu.assert_ref_is_current()` was raising so no GPU job could be
dispatched, and the queue row that ordered the work read **OVERDUE**.

- **Every liveness instrument was green, correctly.** The pacer printed
  `0 failed slots` for 14 consecutive slots and was right: nothing failed. `rc=0`
  is the success code, the PASS count was unchanged at 107 which is legal for
  registration work, and the commits that *did* land were clean. **A monitor
  built on exit codes cannot distinguish "finished" from "stopped partway with
  the receipt already written".** Ask what a half-done success looks like on
  your dashboards; here it looks identical to a good day.
- **The only on-disk trace was an absence**, and it was in an exempt file. The
  missing journal entry was the one artifact that would have said "this slot
  ended mid-unit" — and `LOOP_JOURNAL.md` is in `PROSE_DOCS`, so nothing gates
  on it. The closing step that documents an iteration is the step a dying
  iteration cannot reach, which is `a-rescue-path-gated-on-the-artifact-written-
  LAST-is-unreachable` arriving a second time by a different road.
- **Uncommitted code does not sit still when cron shares the tree.** The usual
  mental model of dirt is passive — work waiting to be inherited. Here
  `regate.sh` fired every ~2 h and *ran* it, so the dirt was authoring ledger
  rows and commit messages. Those 8 messages advertise "the standing-PASS
  rotation", a lane `git show HEAD:scripts/regate.py` does not contain. **When
  an automated writer shares a working tree, uncommitted code is not pending —
  it is in production, and git history will describe behaviour the repo lacks.**
- **The dirt could not clear itself, and every component was individually
  right.** `regate.sh` stages `RUNNER_OUTPUTS` only; `ladder_loop.sh`'s pace-skip
  stages the harvest only. Both correctly refuse `git add -A` because another
  writer shares the tree. So the *only* organ that can commit `scripts/` is an
  unskipped builder iteration — and the pacer skipped 14 straight slots. **Dirt
  introduced by an organ that then loses its own permission to run is permanent
  by construction.** Check that the cleanup path does not depend on the thing
  that broke.
- **The exemption classes were widened twice for docs and never for the
  project's own scripts.** `PROSE_DOCS` grew to cover `LOOP_JOURNAL.md`,
  `LESSONS.md` and `OVERSIGHT.md` after three paid scars (T2.00's stamp
  propagated to 47 specs; 2026-09-02's cross-organ race VOIDed four PASS
  certificates). `scripts/*` is code by the letter of `is_code_dirt` and is named
  by **0 of 257** specs' `impl`/`IMPL_DEPS` — the same "cannot change a verdict"
  argument that justified both widenings, for a class nobody enumerated. **When
  you exempt by listing members, the next member is the one you did not think
  of; exempt by the predicate that earned the exemption.**
- **The repair was already written down and was queued behind the work that
  needed it.** The 150th audit's FTB 5 — *"regate refuses dirty IMPL_DEPS"* —
  sat at item (iii) of the same hand-forward whose item (i) caused this. A guard
  ordered *after* the lane it guards will be reached only on a slot that does not
  need it. **Order a guard before the work it protects, not behind it** — the
  previous lesson in this file says a named-but-untaken repair is a forecast, and
  this is the same forecast coming true one rung lower.
- **HOW IT ENDED, recorded because the ending is the point.** The Sunday FULL
  Review found it unprompted at 06:40 the next morning — in its own `git
  status`, cross-read against `regate.log` — inherited the work, committed it
  (`4a3e646`), verified the lane clause-by-clause against the code rather than
  its docstring, stamped the OVERDUE row ACTED, and routed the structural residue
  as its own row while telling itself in writing that *"the stamp must not
  swallow what committing it repaired."* **No instrument detected this; an organ
  reading a log did.** The repair for the invisibility is still owed, and this is
  the argument for paying it rather than against: the system got 15 hours of
  degraded certificates and a closed GPU path, and was rescued by one desk
  happening to sit on the right morning. That is luck with a good process on top
  of it, not a monitor.

## A counter can mandate an irreversible edit while the state it reads is a two-hour transient (2026-10-19)

The 08:52 regate sweep ran over the 08:07 slot's uncommitted tree and bought
five `+dirty` certificates (VO.01, DP.00, PG.9, NE.00, T0.30). For the two
hours until the stale lane's next clean-tree pass, the blocked-walk treated
those five as "PASS but STALE", and that single cause moved THREE instruments
in OPPOSITE directions at once: `unreachable` rose 96 -> 101 (five specs newly
"out of reach"), `goal_unrunnable` FELL 7 -> 5 (the walk now stopped at the
stale DP.00 before reaching LC.03, so DP.02/DP.03 shed their `welded<-LC.03`
marker and read "live"), and coverage printed a MANDATED edit: *"2
unrunnable-baseline entr(y/ies) are LIVE again and must be removed from
GOAL_UNRUNNABLE_BASELINE: DP.02, DP.03"*. That instruction was correct by the
instrument's own rules and WRONG: the constant is shrink-only ("never add"),
so obeying it during the transient would have moved a ratchet irreversibly on
state that reverts within the hour, and the next honest reading would then be
permanently red with no legal way back.

- **A falling counter is not better news than a rising one; both inherit the
  trust of the state they read.** The fall (`goal_unrunnable` 7 -> 5) looked
  like progress and was the same dirt as the rise. Attribute a delta to its
  CAUSE before classifying its direction as good or bad — here one `git log`
  + one worktree diff named a single cause for all three movements.
- **Before obeying an instrument's mandated edit to a shrink-only constant,
  ask whether the reading it rests on survives a clean tree.** The cheap test:
  does the delta trace to rows stamped `+dirty`, to an uncommitted file, or to
  any state with a mechanical heal already scheduled? If yes, the mandate is
  about the transient, not the project. An irreversible edit obeyed on a
  reversible reading converts noise into permanent record.
- **The dirty-stamp class reaches further than certificates.** Its row priced
  the damage as "a weaker certificate" and one knock-on refusal. Measured
  today: it also drives the blocker walk, the weld classifier, the
  GOAL-citation auditor and the reachability ratchet — every reader of
  `stale_claims` inherits the dirt. The structural guard (regate options
  (i)+(iii), executed this slot) closes the producer; this entry is for the
  CONSUMERS: a reader of a derived counter cannot assume its inputs were
  bought clean.

## An abstention guard keyed on `used + est > slack` is a total refusal once somebody else crosses the line (2026-10-19)

`scripts/regate.py:crossing_a_class_slack` exists for a real scar: on its first
day the clerical re-buy lane spent 4,218.6 s of the day's 4,450.2 s and
foreclosed `cpu<2h` (37 specs, 36 never run) because it consulted no meter. The
repair was to gate every candidate on the **tightest** class's slack, with an
explicit purpose — *"refusing to be the spender that crosses it means the
clerical lane can never again be the reason a science class closed."*

Today the lane declined **64 seconds** of work at 12:43 and printed *"still
stale next tick"*, and the five `+dirty` certificates it declined are the only
things that can return a shrink-only ratchet (`unreachable` 101) to its floor
(95). The reason is arithmetic, not load: the test is
`used + est_s > tightest["slack_s"]`, and `used` (4,197 s) was **already past**
`slack_s` (3,600 s) because the morning's legitimate science spend crossed it.
Once `used > slack`, the test is true for every `est_s >= 0` — the lane refuses
everything until 00:00 UTC, and the day's spend only ever rises.

- **A guard whose trigger is "would this cross the line" becomes "nothing may
  be spent" the moment the line is crossed by anyone else.** The two readings
  are not the same policy, and only the first was designed. When writing an
  abstention test, ask what it does on the far side of its own threshold —
  `crossed_by_me` and `already_crossed` need different branches.
- **The guard's own docstring contained the proof that its refusal was
  pointless**: it states that `used_s > slack_s` *is* "that class has >= 1
  foreclosed member." So by the time it refuses, the class is already closed —
  the 37 specs are unaffordable with or without the 64 s, and the abstention
  protects nothing it was built to protect. A guard that can state its
  trigger's meaning can usually be checked against it; nobody had.
- **A predicted mechanical heal is a prediction, not a repair, and nothing
  re-reads it.** The 10:07 slot correctly diagnosed the dirty window and
  correctly refused coverage's mandated irreversible shrink, then wrote that
  both counters *"revert on the stale lane's first clean-tree pass."* Two
  passes later (10:43 refused on tenant load 15.51, 12:43 refused on budget)
  nothing had reverted, and the 12:07 slot went to sleep **waiting on the
  heal's own watcher** eight minutes before the tick that refused it. If you
  defer a ratchet reading on a predicted heal, name the lane, the next tick,
  and the exit condition — then read that lane's log, not the watcher.
- **The lane that repairs an accident can be foreclosed by the lane that caused
  it.** `W1.02`'s re-buy alone billed 1,138.9 core-s of today's 4,197 s. Science
  spend and clerical spend draw on one day meter, so the cheaper, strictly
  clerical work is always the one that gets refused — and it is the work that
  clears the evidence trail.

## A gate's INPUT can be false, and a monotone meter proves it for free (2026-10-20, 154th audit)

The 90% usage stop refused 15 builder slots, 2 overseer audits and 1 Review poll
in fifteen hours. The policy was correct, the code was correct, the threshold was
correct. The *number* was impossible: inside one unchanged weekly window
(`2026-10-19 05:00 → 2026-10-26 05:00`, from the CLI's own printed reset) the
meter read 7% → 42% → 95% (fifteen hourly reads) → 0% → 2%.

- **A cumulative counter that falls inside its own window has told you it is
  lying, and it costs nothing to ask.** `claude_usage.py` already parses the
  reset instant and already exposes `reset_epoch()`/`--week-elapsed`; every
  ingredient for the check had been sitting in the file for two months. The
  existing lesson here was *"do not model the meter"* — correct, and it was read
  as "the meter is the ground truth." Not modelling a reading is not the same as
  being unable to falsify it.
- **The record did not contain the one field that proves the fault.**
  `usage_ledger()` logs `pct` but not `resets`, so monotonicity-within-a-window
  is uncomputable after the fact. One string, never captured, made an entire
  failure class invisible — and the fix is cheaper than the paragraph describing
  it, because the string is already in the output the function parses.
- **An instrument that only records SUCCESS cannot explain an outage.**
  `usage_ledger()` is called when an organ starts or ends. A gate-refused slot
  starts nothing, so all 18 false readings existed only as English prose in three
  log files. *Log the refusal, not just the run* — the refusal is the event with
  no other witness.
- **Two organs independently explained the impossible reading by assuming the
  meter was honest, twelve hours apart.** The builder saw `93% → 7%` and wrote
  *"off-box movement, read not modelled"*; the Review saw `95% → 0%` and wrote
  *"the usage week has reset and the gate has RELEASED — the blackout's cause is
  gone"*, onto the current-state page. Both are the natural reading. Both
  preserve the instrument's truthfulness at the cost of the finding, and the
  second one **closed** it. When an organ's premise comes from outside the repo,
  "every organ agreed" is not corroboration — it is one unverified reading with
  three signatures.
- **The charitable explanation of an anomaly is the one to write down and then
  attack.** "It reset", "that was someone else's spend", "it self-heals next
  tick" are all unfalsifiable as written and all terminate the investigation. If
  an anomaly is explained by an event (a reset, an external consumer), name the
  event's own observable — here, the printed reset instant — and check it. It
  took one subtraction.
- **Split the repair by who is allowed to make it, or it will not be made.**
  Recording and reporting the invariant is strictly monotone and belongs to the
  builder. "May the organs run when the stop's reading is provably invalid" is
  policy: every version of it either widens what may be spent or invents a second
  limit nobody set, both forbidden to a desk default. Specifying the monotone half
  and routing the policy half is what makes either half happen; specifying the
  whole thing as one item is what makes a desk refuse it for another three days.
- **Say which part of your fix does not fix the thing.** A confirming second read
  before a STOP is correct and cheap, and it would have saved none of these 15
  slots — the reading persisted across 15 reads an hour apart. Letting it ship as
  "the fix for the blackout" would have closed the finding without closing the
  hole.

## A reader keyed on a RENDERING inherits the renderer's flutter — compare the quantity, not the text that denotes it (2026-10-20, 155th audit)

Two of this audit's three worst findings are the same defect in different files,
and neither is in any spec's science. Both are readers that compare *how
something was printed* instead of *what it means*, and both went blind on the
morning they shipped.

- **`METER-NON-MONOTONE`** (shipped 07:19 the same day, to catch the gate's false
  meter) decides "is this the same usage week?" with `resets == prev[2]` — an
  exact comparison of the CLI's printed reset instant. The CLI renders one
  instant two ways: `Oct 26, 4:59am (UTC)` (7 rows) and `Oct 26, 5am (UTC)`
  (5 rows). **5 of 11 consecutive pairs (45%) therefore read as a WINDOW CHANGE
  and are exempt from the invariant**, inside a single real window — and the
  exemption branch is the benign one ("a fall where `resets` CHANGED is the
  legitimate weekly reset"). The repair was one line and the parser it needs
  (`claude_usage.reset_epoch`) was already in the same folder: compare the parsed
  epoch within a tolerance, because rendering flutter is 60 s and real windows
  are 7 days apart — four orders of magnitude of margin, free.
- **`steering._ITEM`** reads the overseer's orders by matching `^N. `, with a
  normaliser for the one drifted style someone had already been caught using
  (`**N. `). This audit's predecessor wrote `**B1.` — a third style — and
  `docs/OVERSIGHT.md` went from **7 parsed items (10-18) to 6 (10-19) to 0
  (10-20)**. The same file went dark the same way on 10-10. The orders were still
  executed, because the builder reads prose; what was lost silently is the
  legality check that tells a reader an order names a BLOCKED spec.

The generalisable shape, and why it keeps recurring here: **a rendering is a
lossy, unversioned, externally-owned encoding of a value, and treating it as the
value makes an equality test depend on a formatting decision nobody in this repo
controls.** The failure is always in the safe-looking direction — two things that
ARE equal read as different, so the check declines to fire and prints `[]` or
`ZERO items`, which a reader cannot distinguish from "checked and clean".

Three practices that follow:

- **If a value has a parser anywhere in the repo, the comparison must go through
  it.** `reset_epoch` existed and was used elsewhere for pacing; the new reading
  re-derived "same window" from scratch and got it wrong. Searching for an
  existing parser is cheaper than writing a comparison.
- **House style is not a contract unless a test holds it.** Three organs write
  `FOR THE BUILDER` and each remembers a different numbering convention; the
  parser comment had already *predicted in writing* that a third style would go
  dark. A convention only one of three authors can recall is a latent outage with
  a note attached, and the note does not help.
- **A zero-count reading needs a loudness that a true zero does not trigger.**
  `page_stats` was built after an earlier five-day outage to make "heading
  present, zero items parsed" LOUD, and that loudness is the only reason this was
  caught. `meter_non_monotone` has no such distinction: a genuinely monotone week
  and a week whose pairs were all exempted both print `[]`. When a reading can be
  empty for two very different reasons, print which one.

## An ownership test that never asks whether the owner is alive (2026-10-20, 156th audit)

`coverage.fail_unowned` exists so that a settled FAIL cannot sit with nobody
assigned to repair it. It reads **0** and sits at its declared floor. Six
settled FAILs — including `W.1` and `W.2`, which encode `GOAL.md`'s *"too cold
kills him"* and *"the needs ARE the curriculum"* — are credited to queue rows
that are **all `ACTED` or `DECLINED`**. A terminal row is never re-read by
anyone. The work is owned by nobody and the instrument says it is owned.

The mechanism is one missing conjunct: `_owned_by_dued_row`
(`coverage.py:1352-1417`) walks every `ROUTED:` block and credits ownership the
moment it finds a `DUE:` declaration, with **no filter on the row's status**. A
row's historical `DUE:` line survives its disposition — correctly, because this
file never deletes history — so a dated promise made in September still reads as
a live assignment in October, forever.

The generalisable shape: **a reference is not a relationship.** Checking that
*something points at X* is cheap and checks almost nothing; checking that *the
thing pointing at X can still act* is the property you actually wanted. This
project has now found the same defect in three organs under three names —
`review_queue`'s `DISPOSITION-ON-A-CLOSED-DECISION` (100th audit B2: an `ACTED`
row re-parents work onto a decision that already closed),
`WAITS-ON-A-CLOSED-WINDOW` (a live row waiting on a terminal one), and this.
Two of the three *print* the pair; the third silently counts it as health.

Three practices that follow:

- **When one organ publishes a liveness predicate, every organ that reasons
  about the same objects must import it, not re-derive it.**
  `review_queue.LIVE = ("OPEN","HELD","DISPOSITIONED")` already existed, and
  `_owned_by_dued_row` already imports that module for `_DECL` — for exactly the
  stated reason that two readers must not drift on what a clock looks like. The
  same argument applies to what a *live row* is, and was not made.
- **A counter whose healthy value is 0 needs its credit path audited, not its
  alarm path.** Everyone tests that the alarm fires on a naked FAIL; the fixture
  here does (`Z.1`). Nobody tested that the alarm *stops being suppressed* when
  the suppressor dies, because that requires a fixture of a row changing status
  over time, which is the one shape a static fixture does not naturally express.
- **The repair raises the number, and that is the point.** Fixing this moves
  `FAIL-UNOWNED` off its floor. The shrink-only ratchet is what makes that
  honest rather than embarrassing: raise the baseline in the same commit with
  the reason in the growth log. A repair that lowers its own number is the
  failure mode `T0.31` was built for; a repair that *raises* one and says why is
  the system working.

## The route must precede the prose (2026-10-21, 158th audit)

The ruling on `oversight-for-the-builder-has-no-reader` was executed into
`scripts/overseer_prompt.md` on the night of 2026-10-20 (`c8ee77b`). It says
what it should say: an `OVERSIGHT.md` item whose discharge condition cannot be
evaluated before the next audit rewrites the page must be routed as a `ROUTED:`
row in `docs/REVIEW_QUEUE.md`, because the page is overwritten at 00:37, 06:37
and 12:37.

The next morning, the first audit under that rule found a real defect (a GPU
job's `ok=False` label frozen by `Budget.charge`'s dedupe guard, hiding the only
productive GPU hour of the week), wrote it up as its leading finding, wrote the
sentence *"Routed as `gpu-charge-ok-flag-is-frozen-by-the-dedupe-guard` … and a
`ROUTED:` row"* — and then **exited `rc=124` before appending the row.** The
page was sealed as a draft and the slug appears nowhere in
`docs/REVIEW_QUEUE.md`. The next audit was the 12:37 rewrite the sentence itself
predicted.

**So the ruling was obeyed in intent and defeated by order of operations.** And
the defeat is worse than simply losing the finding, because the page asserts the
row exists: a reader auditing whether the item was tracked finds a sentence
saying yes, and stops looking. A false claim of durability is strictly worse
than an admitted gap — an admitted gap gets re-derived, a false claim gets
quoted.

The generalisable shape: **when a finding must outlive the document describing
it, the durable write is the first act, not the last.** Prose about a durable
record is not the record. And the ordering is not a style preference, it is the
only version that survives a timeout — a run killed mid-sitting leaves either
(a) a row with no prose, which the next reader finds and acts on, or (b) prose
with no row, which the next reader finds and *believes*. There is no third
outcome worth trading for, because the expensive half (deciding that the finding
matters) is already done by the time either write happens.

Three practices that follow:

- **Append the `ROUTED:` row, then write the paragraph that cites it.** The
  paragraph can then quote the row's real slug and `DUE:` rather than promising
  them, which is also a better paragraph.
- **A seal cannot repair this and should not try.** `lib_seal.sh` correctly
  banners the page as an incomplete draft, and that banner is what let this be
  found — but a banner over a page that asserts a route still asserts the route.
  The seal marks the *page* untrusted; it cannot mark one sentence inside it
  false. Only ordering can.
- **Audit your predecessor's receipts, not only its reasoning.** The existing
  lesson *"Audit the predecessor's reason for not acting"* covers a wrong reason
  that vanishes with the page. This is its sibling: a **claimed act** that
  vanishes with the page. Both are found the same way — re-derive the fact, and
  grep for the artefact the claim names.

Found in the same sitting, and recorded here as the second mile of *"An
ownership test that never asks whether the owner is alive"* above rather than as
a new lesson: making `fail_unowned`'s credit path status-aware was the right
repair and it is still too generous. It credits a row as a FAIL's `queue-row`
owner — the strongest queue form, *"a row ABOUT the spec … carries a `DUE:`, a
dated promise"* — on any appearance of the id in the row's live extent. `SO.10`
was therefore owned by a row about a venue comparator that merely cited *"the
`SO.10` vacancy precedent"*, and 11 of the 26 credited FAILs (42%) are held up
the same way. **A reference is not a relationship — and a citation is not even a
reference to the same subject.**

## A freshness claim is a claim, and it is the one nobody checks (159th audit, 2026-10-22)

`D47` carries a section headed **"The measurement, read this sitting"** and
then quotes four numbers. Re-derived with the instrument's own
`review_queue.throughput()` against every October revision of its own subject
file: the fresh breakdown (`39 OPEN, 35 DISPOSITIONED, 0 HELD`) was read that
sitting and is exactly right, while the **headline pair** (`77 live rows`,
`drain 54 cycles`) is a reading from **four days earlier**, and a third pair
(`arrivals 2/cycle against disposals 1.71/cycle`) was never printed by the
instrument at any revision in two months. The entry's own breakdown sums to 74
while its headline says 77, and its `DECIDE:` block disagrees with its body.

**Why this is worse than an ordinary stale number: a clock was attached to it.**
The armed default's stated price was *"the drain reading goes on printing 54
cycles"*; the live reading was **101**. A default fires the day after
`decide_by` and the overseer fires on the first legal day, so the firing would
have restated — as the cost of holding — a figure understating it by ~2x. The
pre-registration discipline this project runs on assumes the registered price is
the measured one. **An armed default inherits the staleness of every number used
to price it, and firing it launders that staleness into a ruling.**

The generalisable shape: **a sentence asserting that a number is fresh is
evidence of nothing, and the organs here have no reader for it.** `T0.22`
already forbids this exact shape for SPECS — *"A number borrowed from another
spec's entry cannot be stale"* — and the decision register, the queue and the
desk pages have no equivalent. So the one place where staleness is explicitly
gated is the place it is least likely to occur.

Three practices that follow:

- **Re-derive a quoted instrument reading by calling the instrument, not by
  reading a later page.** Every reading in this repo is a pure function of
  files that git already has, so "what did this print on 10-17?" is answerable
  exactly. There is no need to argue about a number that can be recomputed.
- **Validate the reconstruction before trusting it: reproduce TODAY's printed
  reading first.** The method here was only worth anything because it returned
  `arrived 16 / disposed 21 / live 72 / drain 100.8` against the tool's printed
  `101`. A reconstruction that has not been checked against a known-correct
  output is a second opinion with no credentials, and it would have been very
  easy to "discover" a discrepancy that was merely my own wrong baseline.
- **Check whether the quoted figures are consistent with each other and with
  the formula.** `arrivals 2/c` above `disposals 1.71/c` makes `net >= 0`, which
  `review_queue.py:510` renders as `unbounded` — so that pair and a finite
  `drain` cannot have come from one reading. **An entry can be convicted of
  staleness from the inside, before any git archaeology**, and that check costs
  one reading of the source.

And the direction worth noting, because it was not symmetric: **every stale
figure here was the rosier one.** Numbers do not go stale at random when a desk
is arguing a case — the ones that get re-measured are the ones whose movement
would change the argument, so the surviving quotes drift toward whichever
reading was most quotable. That is not dishonesty and it needs no bad actor;
it is what happens when a page is assembled from earlier pages.

Sibling of *"Overseer prompt quotes stale numbers"* (every number in the
instructions is a dated example) and of *"Gate inputs can be false"* (a weekly
pct falling inside its own reset window). Those two are about inputs arriving
wrong. This one is about a number that was **right when written**, quoted later
under a claim of freshness, with a deadline hanging off it.

## A control that certifies an absence is only as good as the last time the presence was seen (2026-10-25, builder, executing the `t310-anticorrelated-gates` ruling of 2026-10-01)

`T3.10` certifies "the freeze held" by contrast: the unfrozen arm must drift
(`probe_drift_unfrozen >= 0.10`) so that the frozen arm's near-zero means the
freeze did something. That is a NEGATIVE control — it certifies an absence by
comparison with a presence. The spec's two repair pilots then measured the
trade exactly: at `EPOCHS_P` 40 the contrast is alive (drift 0.1875) and the
claim is unsatisfiable by arithmetic; at `EPOCHS_P` 150 the claim becomes
measurable (`knowledge_margin_min` 0.0299) and the contrast vanishes (drift
0.0078) — because a converged trunk is one whose features phase A's gradients
no longer move, *frozen or not*. **There is no value of the knob that fixes
it**: under-training restores the presence and unmeasures the claim,
over-training measures the claim and removes the presence. The apparatus
change (convergence) silently expired the control's credentials, and the spec
had no way to notice.

**Rule (the ruling's, generalised):** a control that certifies an absence is
only as good as the last time the corresponding presence was seen; if the
apparatus changes, re-demonstrate the presence or the absence stops being
evidence. Mechanically: a zero-reading claim (zero drift, zero leak, zero
trips) is admissible only on a run that ALSO demonstrates the channel can fire
at the SAME operating point — a pre-registered calibration arm whose
perturbation is declared before the run. A calibration arm that cannot fire is
a VOID of the instrument, never a FAIL of the claim, and never a tuning
opportunity. This is the second instance of the class — `lt02-the-venue-has-
no-true-positive-body-chaos-is-reducible` is the same defect through a venue
channel: **a measurement whose failing side has never been observed to be
observable.**

The permanent guard, per the ruling: the requirement attaches to the
frozen-vs-plastic FAMILY, registered in `docs/CHAMPIONS.md` beside the
PLASTIC-ONLY decree (arenas `PL.00`/`PL.02`) — the next registered spec that
reads a freeze-held claim carries the calibration arm. `T3.10` itself stays
PARKED; the one-diagnostic cap is SPENT; no bar moved in either direction.


## A proof that a mechanism is safe in ONE direction leaves the other direction with LESS scrutiny than it had before

Two instances, found in one sitting (161st audit), in two files, with nothing in
common except their shape — which is why the shape is worth writing down.

**Instance A — a one-sided safety proof.** `experiments/cpu_budget.py`'s
admission projection carries an explicit proof of harmlessness: *"Clamped at the
enum by construction, so the projection may only TIGHTEN an estimate and this
gate can never refuse something it would have admitted before,"* and `T0.33`
property 12 asserts it over the whole registry. The scar recorded in the same
file points the same way — `SO.08` sat foreclosed a full day at ~28,000× its
measured cost. **Both the proof and the scar are about FALSE REFUSALS.** Read
live today, the same function returns `(10.12, "MEASURED 0.03s x4 + 10s")` for a
spec whose run cost 51,841 core-seconds — roughly 5,100× too LOW, under a
provenance string asserting a measurement. Nothing had looked, because the file
already said the function was proved safe.

**Instance B — a repaired write, an unexamined erase.** `lost_iterations.log`
exists so that *"a dead iteration must be a number, not a silence"*
(`ladder_loop.sh:370-371`). Its WRITE half failed through an entire outage, was
diagnosed, and was repaired — and the repair's reasoning sits in the file as a
comment, which is exactly what makes the file look audited. Eleven lines later
the same script truncates that file on the first `rc=0` (`:408`, `: > "$LOST"`).
The erase destroys the record the repair restored, and three slots lost the same
day existed nowhere tracked once the loop recovered.

**THE GENERALISATION, and it is an instruction for reading, not for coding.**
A mechanism that carries a proof, a scar or a comment about one of its directions
is *harder* to audit than one that carries nothing, because the artifact of past
scrutiny reads as current coverage. So when you find one, do not check the
direction it names — check the one it does not, and ask two questions in order:

1. **Which direction did the proof examine?** Write the sentence out. "May only
   tighten" is a statement about refusals and says nothing about admissions.
   "The marker is written" is a statement about creation and says nothing about
   destruction.
2. **What does the unexamined direction COST, and is the cost recoverable?**
   Rank by cost, never by symmetry. The two directions of an estimate error here
   are *a dispatch this project did not get* and *a shared box lent out at three
   times the metered rate* — and SYSTEM.md law 4 puts the second above the
   ladder. A false refusal is visible the moment it happens and costs an
   iteration. A false admission is invisible by construction and spends somebody
   else's machine.

The existing lesson *"One direction alone is worthless, because a guard that
refuses everything and a guard that refuses nothing leave the same clean log
behind them"* (`T0.18`'s two-sided fixture) is the same principle applied to a
GATE. This is the same principle applied to a **proof about a gate** — and it is
the harder case, because the gate's log is at least empty, while the proof's
prose is full.

## Ask whether a safeguard reads the RUN or the RECORD — one of those can only
## confirm damage; and a rule that shields a weak certificate from a wrong
## demotion also shields it from measurement

*(162nd audit, 2026-10-26, from two findings in one sitting that turned out to
be the same shape.)*

**THE FIRST HALF — a safeguard that reads the record is a post-mortem wearing a
guard's clothes.** `aggregate-hides-worst-seed` was ruled *(c) THEN (b)* on
2026-09-30: arm (c) the static audit, arm (b) the recorder refusal that the
ruling itself calls *"the only arm that makes the bug unrepeatable."* Arm (c)
shipped the next day and was treated thereafter as the class's coverage. On
2026-10-25 the class produced its first confirmed false PASS — `WM.01` attempt 1
recorded PASS on a **mean** margin of 0.1927 over per-seed margins
{0.4707, 0.0800, 0.0273}, against a bar its own docstring pre-registered as *"on
every seed"* — and arm (c) could not have stopped it:

- its **LANE A** fires only when the gated key is produced by a `min`/`max`/`len`
  fold in the same file (`worst_seed_audit.py:521`, `if g.key in folds`), and the
  defective key was a bare subtraction;
- its **LANE B** would have fired — `mu - sigma*sqrt(2)` lands at −0.087 against
  a bar of 0.10 — but LANE B **reads the committed ledger**, so it cannot speak
  until the wrong verdict is already recorded.

So the class was undetectable before the verdict and detectable only after it,
and twenty-five days of "arm (c) is shipped" read as coverage of both.

**THE QUESTION TO ASK, in this order.** (1) *Does this safeguard read the run or
the record?* A record-reader is a detector, never a preventer, and the two are
not interchangeable however good the detector is. (2) *If it reads the record,
what has to go wrong before it can speak?* Write that sentence down next to the
safeguard. For LANE B it is "a false PASS has to exist." (3) *What invokes it?*
`run worst-seed-audit` is in no crontab entry, no script, no prompt order, not in
`run status`, has no ratchet counter and always exits 0 — a detector nothing
calls is a detector that has not been built, and the difference between the two
reaches no exit code.

**THE SECOND HALF, and it is the same shape from the other side — a protection
and an exemption can be one mechanism, and only one of them gets counted.**
`pass_on_dead_dependency` reads 7 against a declared floor of 3. The regate
rotation, the only organ that routinely re-buys a standing certificate, refuses
four of those seven every sweep — *"rotation would record BLOCKED, not a
verdict"* — which is the **correct** refusal: demoting a PASS by bookkeeping
rather than by measurement is strictly worse than leaving it. But the same
refusal is why `T0.19`'s PASS has not actually run since 2026-08-11. The counter
records the weak foundation and nothing records the exemption, so the board shows
*"7 certificates on dead dependencies"* and never *"and none of them can be
re-measured."* Three of the seven hold champion seats, which `coverage` cannot
see, while `champions --check` cannot see the dependency death.

**THE GENERALISATION.** When a mechanism protects something you distrust, ask
what else that protection withdraws. The honest repair is almost never to remove
the protection — it is to make the pair printable, because a number that counts
the danger while hiding the immunity reads *safer than nothing at all*. This is
the same disease as the 161st audit's finding that both recorders of a lost
builder slot erase themselves on recovery, and as the lesson above it about a
proof that examined only the direction that costs a dispatch: **every counter in
this project that would show it doing harm either resets when the harm ends, is
blind until the harm is finished, or was never built.** That sentence is now
true of three independent mechanisms measured inside forty-eight hours, which
makes it a property of how this project builds instruments rather than three
coincidences.

## A ratchet that may shrink has to be TOLD when it shrinks, or the shrink becomes headroom — and a SET-shaped baseline keeps exempting the defect that healed

**The measurement (163rd audit, 2026-10-26).** Two commits six hours apart did
unambiguously good work: `90acbbf` executed `SM.03`'s arm 3 with a *new* gated
conjunct and nine bars byte-unmoved, and `71e5b96` re-classed its budget
`GPU_SHORT -> CPU_LONG` because the runner had been correctly refusing to
dispatch a CPU-bound spec that declared GPU. Between them they moved two ratchet
counters in the *allowed* direction — `claim_dead` **3 -> 2** (the `smell`
commitment regained a live falsifiable path) and `park_release_pairs` **3 -> 2**
(`SM.02->SM.03` became a walkable release). `git show --stat` on each: one file,
one file. **Neither ran `run ratchets record`.**

Nothing was falsified and no bar moved, so every instinct says this is
housekeeping. It is not, for two separate reasons.

**1. An unbanked honest shrink is stored slack.** The floor stays where it was
while the live reading falls below it, so the gap is headroom the next
regression spends silently. The counter will read `AT floor — ok` all the way
back up to the old value. The rule printed beside every moved counter —
*"if a committed change justifies it, `run ratchets record` in that commit"* —
is usually read as a permission. It is an **obligation**, and it binds hardest
in exactly the case that feels least urgent: when the movement is good news.

**2. A `frozenset` baseline is worse than a number, because its stale member is
an exemption rather than a gap.** `PARK_RELEASE_BASELINE` (`coverage.py:579`)
still names `SM.02->SM.03` after that pair healed, and the tool's own closing
line is an instruction, not a reading: *"1 baseline pair(s) are WALKABLE again
and must be removed."* Set membership silences the member. `SM.03` carries a
**pre-registered STOP** that can withdraw arm 3 — so the documented exit
condition of the very run this edit enabled would return the pair to unwalkable,
where it reads *baseline-blessed* and never banners again. The commitment would
go claim-dead a second time behind a green check.

**THE GENERALISATION.** `T0.31` and its siblings watch for a count lowered
*dishonestly* — the tidy-up that quiets an instrument. This is the mirror image
and no gate in the project looks at it: a count lowered **honestly** and not
written down. The asymmetry is the lesson. A dishonest shrink is an act someone
chose and can be caught in a diff; an honest shrink left unbanked is an
**omission inside a good commit**, which is the one place this project's
reviewers reliably stop reading. So the audit question is not only *"did a
number fall for a bad reason?"* but *"did a number fall for a good reason and
nobody tell the ratchet?"* — and for any set-shaped baseline, *"does it still
name something that has recovered?"*, because there the cost is not a missing
record but a standing immunity. This sits directly beside the lesson one shelf
up: a counter that hides an immunity reads safer than nothing at all. Here the
immunity is granted by a line of bookkeeping nobody owed anyone.
