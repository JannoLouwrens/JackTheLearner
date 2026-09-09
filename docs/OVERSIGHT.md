# OVERSIGHT — 86th audit, 2026-09-09 06:37–07:2x UTC (at `f7d7d5d`, tree clean)

## VERDICT: DRIFTING — **the ledger is sound and nothing was loosened, but the builder has been dark for 22 hours, the demonstrated set has not moved in two days, and W37's entire 30-hour free GPU allocation is on course to expire unspent for the third time in six weeks. The Review found the cause six minutes into this audit and routed it as `D26`. My job today is the half of it `D26` does not price: the builder's own model meter reads 95%, the loop's `MODEL_FLOOR` is 95, and 72% of that meter is not ours either.**

**This audit ran concurrently with the Review** (`37 */6` and `37 6` collide at
06:37). The Review committed `f7d7d5d` at 06:43 while I was mid-sweep. I had
independently reached the same diagnosis from a different direction — the
diurnal shape of the meter and the `ps` table — before reading `D26`. Two organs
converging on one finding from two evidence bases in the same ten minutes is
worth recording as a fact about the system, not just about the finding. **I have
not re-routed it.** Everything below either confirms `D26` against my own
reading or adds what it does not carry.

Sections 1, 2, 6 and 7 are clean and that goes on the record before the findings.

---

## RANK 1 — `D26` is right, and its recommended option is priced on the wrong model

`D26` (Review, 06:43 today, `decide_by` **2026-09-10**) establishes that
`pace_gate` starved the builder on a meter that is 62% somebody else's. I
confirm every load-bearing number in it independently:

| claim | my reading |
|---|---|
| builder dark since `2026-09-08T08:23` | confirmed — 22 consecutive `PACING:` lines in `ladder.log`, 09-08 09:07 → 09-09 06:07, zero iterations |
| `week:all models` 59%, line 45%, elapsed 30% | confirmed live at 06:5x: `claude_usage.py --pct` = **59**, `--week-elapsed` = **30**, `allow = 25 + ceil(65·30/100)` = **45** |
| the spend is not ours | confirmed by a second route: `usage_ledger.jsonl` has **no entry at all** between `2026-09-08T08:23` and `2026-09-09T06:37`, across which the meter rose **31 → 59 (+28 points)**. The rise is **diurnal** — climbing 09:07→00:07, flat 00:07→06:07 — which is not the shape of reporting lag from one 16-minute Fable job. `ps` shows two long-lived non-organ `claude` processes owned by `opc`, one a `--fork-session --resume` at `--effort xhigh --permission-mode bypassPermissions` rooted in `/home/opc/.claude/projects/-home-opc/` |
| forecast wake `2026-09-10T17:23`, 57 dark hours total | confirmed by arithmetic: the line rises 0.65 pt per 1% of week ≈ **9.3 pts/day**; the meter rose ≈ **30 pts/day**. The gap widens, it does not close. The forecast holds only if the external draw stops |

### What `D26` does not carry, and the owner needs it before ruling tomorrow

**`D26`'s option (i) says "Effect today: 23% own-spend against a 44% line — the
builder resumes this hour." The builder would resume, but not on Fable, and
`D26` prices it on Fable.**

`crontab` runs the loop as `JACK_LOOP_MODEL=fable`. `lib_usage.sh:181` sets
`MODEL_FLOOR=95` and `model_gate` refuses at `mpct >= 95` (D14 option (b),
effective reading — the loop has fired this refusal 85 times, most recently
through 2026-09-04). **`week:Fable` reads 95 right now.** Same attribution
method as `D26`'s, applied to the model meter:

    week reset 2026-09-07 05:00 UTC; Fable meter read 0 at the first builder run
    builder Fable consumption, summed over 28 start/end pairs        27 points
    rise while the builder was not running                           68 points
    ------------------------------------------------------------------------
    week:Fable at 2026-09-09 06:37                                   95%

**72% of the builder's own model meter was spent by something that is not the
builder**, and 48 of those points arrived in the 22 hours since it last ran.
The builder's final iteration ended at Fable 47%.

So under (i) the loop clears `pace_gate`, hits `model_chain`, is refused Fable
at the floor, and walks to **Opus** (`FALLBACK_MODELS="opus sonnet"`). It runs
— but every iteration is then an Opus iteration billed against the shared
all-models meter that (i) has just stopped gating, and `D26`'s "23% own-spend"
is computed from a history that is almost entirely Fable slots. **I am not
saying (i) is wrong. I think it is probably right.** I am saying its stated
effect is a Fable price for an Opus outcome, and a gate decision should not be
made on that. Routed as an evidence addendum on `D26` itself, not as a new
entry.

### The cost, as a number

`gpu_budget.json` has **no `2026-W37` key: 0.00 of 30 free Kaggle GPU-hours
charged this week.** They expire **Sunday 2026-09-13**; the Claude week resets
**2026-09-14 05:00**. The builder wakes 09-10T17:23 at the earliest.

This is the third occurrence of the failure `pace_gate` was built to prevent,
and `lib_usage.sh:36-43` records the first two in its own comment:

    W32   dark ~4.5 d   8.82 of 30 GPU-h expired unspent
    W33   dark ~2.7 d  22.11 of 30 GPU-h expired unspent
    W37   dark 57 h    30.00 of 30 at risk        <- this week

## RANK 2 — no instrument in this repo can see a builder that is alive and producing nothing

This is mine, it is distinct from `D26`, and it is why a 22-hour outage was
found by hand rather than printed.

- `grep -rl PACING experiments/ scripts/` returns **only** `lib_usage.sh` (the
  emitter) and `ladder_prompt.md`. Nothing counts consecutive skipped slots.
- `lib_liveness.sh:table_liveness` asserts on **history-row dates and file
  ages**. A `PACING:` skip appends to `ladder.log` every hour, so the builder's
  liveness reads green throughout.
- The proof it misleads: the Review's own 09-08 page reported *"builder
  **06:11** (hourly)"* under "verified against `/data/jack-logs` mtimes rather
  than anyone's report" — on the morning the builder was two hours from going
  dark for 57. The verification method was honest and the answer was wrong.
- `status`'s RATCHET COUNTERS block covers queue, champions and coverage. There
  is no counter for builder output.

`D26`'s default (iv) proposes exactly this counter — and **fires 2026-09-10,
after the builder is forecast to wake.** The remedy arrives after the
occurrence it would have caught. That is not an argument against (iv); it is an
argument for the builder implementing the counting half now, which costs
nothing and gates nothing (**B2**).

This is `LESSONS.md`'s standing shape: no organ watches for the *absence* of a
result.

## RANK 3 — `D22` is OVERDUE and the pre-registered default is due to fire

`decide_by: 2026-09-08` passed unanswered. `decisions --check` prints
`D22  costs 0 specs  OVERDUE — DEFAULT IS DUE TO FIRE`.

**The owner did not rule by 2026-09-08, so the pre-registered default fired:**
**(i) THE RULE STANDS** — design authority stays with the Review, unchanged and
unnarrowed. Nothing is written, nothing is re-parented, no threshold moves, no
control weakens, `GOAL.md` is not touched, no commitment goes claim-dead.

The firing is paperwork and is also, on today's evidence, the substantively
right outcome: the Review's own 09-08 `FOR THE OWNER` item 1 said its
recommendation was *"unchanged in substance and weaker in confidence"* after
its best-ever morning, and asked the owner to **wait a week and re-measure on
09-15** rather than grant the ask. The default and the author's current
preference agree.

**Reversal:** the owner may rule (ii) or (iii) at any later date at no cost; the
default wrote nothing. **The deadline was NOT extended** — a deadline that moves
when it is reached is the deadlock the armed-default mechanism replaced.

I have appended the OVERDUE NOTICE to `DECISIONS_NEEDED.md` with the required
wording. Per the `D17` precedent (2026-09-08) and `D13` — the overseer does not
write `DECISIONS_RESOLVED.md` — the resolution record is **B1** for the builder.

## RANK 4 — the two standing `coverage` reds are routed, and both fall due into a week the builder cannot work

`coverage` EXIT 2, unchanged in composition:

- **`CITED-BUT-UNRUNNABLE`, 4 NEW: `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`** —
  all `welded<-LC.07`, which is `PILOT-BLOCKED`. `GOAL.md:193-195` cites all
  four in the present tense as what generality *is*. Routed as
  `goal-cites-four-specs-that-resolve-to-corpses`, **DUE tomorrow, 09-10**,
  OPEN 7 days. `GOAL_UNRUNNABLE_BASELINE` remains `{DP.02, DP.03, LC.04}` and
  must not grow.
- **4 CLAIM-DEAD commitments** — smell, balance, shelter/building, thermal
  (kills). Every claim spec parked or foreclosed; three of the four are owner-
  named constitutional commitments (*"too cold kills him"*, *"every sense a
  human has"*, the owner's own image of success). Routed as
  `five-commitments-are-claim-dead-behind-foreclosures`, DUE 09-11. **The row's
  own title is now stale by one** — the tool reads four, not five. A row that
  names a count should carry the count it was routed at or be re-titled; this
  one silently disagrees with the instrument it exists to discharge (**B3**).

Both dates land inside the dark window. Neither can be paid by the builder
before it wakes.

## RANK 5 — the queue: zero violations, and the pile is now scheduled to break

`run review-queue` EXIT 0, **0 violations** — correct and not reassuring.

- **41 live rows** (27 OPEN, 2 HELD, 12 DISPOSITIONED); oldest live 16 d.
- **Trailing 7 cycles: arrived 26, disposed 3, designed 12. `drain UNBOUNDED`.**
  A `DISPOSITIONED` row is designed, not disposed — it still ages.
- **8 rows due TODAY** and **10 rows due 2026-09-13** against a measured
  one-cycle maximum of 6. Ten promises are scheduled to break together, on a
  Sunday, in a week whose builder is offline.
- `review_queue_net_arrivals` **29 → 23 (!! MOVED −6)**. No committed change
  justifies it; it is the trailing window sliding. The Review was mid-run when I
  read it and should record it in this morning's page.

## SECTION 1 — integrity of the ledger: CLEAN

- **108 PASS rows. 108/108 resolve to an implementation** with a `run()` via
  `_module_for`. **108/108 recorded commits exist in git** (`git cat-file -e`).
  **108/108 specs declare a `control`.** **0 PASS rows** lack a
  control/null/shuffled/twin token anywhere in the recorded row.
- `audit_supersedes_fail`: **3 live violations, 11 checked pairs, 23
  unauditable.** All three are the known `+dirty` class — an adverse verdict
  stamped at an uncommitted implementation whose bytes were never preserved:
  `LG.00` VOID `8faff43+dirty` (08-30), `T0.29` FAIL `661a48f+dirty` (09-02),
  `T0.29` FAIL `44e54a7+dirty` (09-06). This is `T0.27`, held deliberately RED
  by `D16`'s fired default *"option (b) ALONE — the warning stands, `T0.27`
  stays RED and is not touched"*. Unchanged since 2026-09-07. **A deliberate red
  is not a clean bill; it is a debt with a decision behind it, and it is still
  three.**

## SECTION 2 — thresholds and controls over 7 days: NO LOOSENING, one edit that deserved the look

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`.

- **One numeric constant moved: `T0.31 N_PROPERTIES 16 → 17`** (`021d302`) —
  strengthening. No control deleted, no seed count reduced, no assertion
  removed. New `PL.02` constants are registrations of new bars, and
  `EYE_RADIUS_R2_MIN 0.80` is carried from `PG.6` unmoved.
- **The one hit worth reporting, because §2 names `_check` gaining an `or`.**
  `cd5a27b` changed `T0.31` P14 from
  `pile["due_pile"].get(free, 0) != 0` to `... >= CAP`. That is a **weakening
  of what the test demands**: the property used to require `next_free_due` to
  return an *empty* day and now accepts one carrying up to `CAP-1` rows.
  **I judge it correct and not a violation**, because it is definitional — with
  the old `CAP = 1` the two forms are identical, and `next_free_due` is
  specified as "the next date under the measured capacity", not "the next empty
  date", so the old form would have failed a correct tool once the constant
  moved. The commit rebuilt the fixtures *from* the constant so the battery
  certifies the mechanism at any honest value. Disclosed in the commit message.
  **Reported here rather than waved through, because a loosening-shaped edit in
  the same commit that raises the constant it depends on is precisely the pair
  §2 exists to make someone else look at.**
- **A follow-on concern about that constant, which is not a violation.**
  `MEASURED_DISCHARGE_CAPACITY 1 → 6` was raised on **one** cycle — the Review's
  best morning ever — while `throughput()` reads **0.43 disposed/cycle
  sustained**, a 14× gap. The builder recorded the caveat in the docstring
  honestly. But the AMBER pile warning still *divides by 6*, and
  `review_queue_piled_on` fell **26 → 6** in the same commit (recorded
  DEFINITIONAL) and the flagged dates fell 6 → 2. An instrument got quieter on
  the strength of a single record-setting day. See **B4** — print both readings;
  do not move the ratchet.

## SECTION 3 — drift: none in what was done; a real hole in what was not

**Last day's builder work** (three iterations, all on 09-08): the 85th audit's
items 1–4 — `MEASURED_DISCHARGE_CAPACITY 1 → 6`, `kindless_arena_discharges`
counted at 9 then migrated to 1, the day-scoped banner's `today` moved to
`time.gmtime` — plus `T0.29`/`T0.31` certificate re-buys.

**Which GOAL.md sentence it serves:** the first-principle block, *"or protects
the honesty of watching what happens when the three meet."* That clause is real
and this is squarely inside it. **Not drift.**

**The converse, which is the harder question and the worse answer.** The
demonstrated set has not moved since **2026-09-07** (108/245). The last three
writes to `ledger.json` are all Tier-0 self-certification — the ladder
measuring its own instruments. Against `GOAL.md`'s named claims:

| commitment | specs | passing |
|---|---|---|
| one brain / unison | 27 | **1** |
| curiosity | 12 | 2 |
| hearing | 14 | 1 |
| fast/slow | 8 | **0** (5 welded behind `LC.03`) |
| sleep | 5 | 0 |
| smell · balance · shelter · thermal | 11 | **0 — CLAIM-DEAD** |

Curiosity, all-senses fusion and learning-by-living are the three my brief names
as most likely to be quietly neglected. They are 2/12, 1/27 and 0/11.

## SECTION 4 — is the builder alive and productive: ALIVE, NOT PRODUCTIVE

**3 iterations in the last 24 h** (09-08 06:11, 07:17, 08:23), **3 × rc=0**,
**PASS delta 0** (108 → 108). **22 iterations skipped, 0 attempted, since
09-08T08:23.** No crash, no credit exhaustion, no aborted load, nothing paused
by a human. The loop is executing its own pacing rule exactly as written. See
RANK 1 and RANK 2. `lost_iterations.log` is 0 bytes; `declared_pids` clean; no
stray processes attributable to this project.

## SECTION 5 — compute honesty: nothing wasted, everything unspent

- **`2026-W37`: 0.00 of 30 free Kaggle GPU-hours charged.** No key in
  `gpu_budget.json` at all. Quota expires Sunday 09-13.
- `gpu_hours_no_verdict` TOTAL **48.07 h**, dominated by **`D1.0` 33.78 h across
  2 attempts and 0 verdicts**. That is correctly frozen: the Review's own FTB
  forbids a third `D1.0` dispatch until the twin-spread probe result is on the
  row and the gate is committed, and then only into W37. **W37 is now the week,
  and the organ that would dispatch it is offline.**
- `gpu_unattributed_jobs` **21, at its declared floor** (6.32 h). Shrink-only,
  not moving.
- The waste this week is not spent hours with nothing to show. It is **30 free
  hours with nothing dispatched into them.**

## SECTION 6 — stuck decisions: CLEAN, with two clocks running

- **`MEANS-ESCALATED`: none.** No fork that a measurement could settle is on the
  owner's desk. The `D1` disease is not present.
- **`UNDECLARED`: 0 of 10.** Every open decision carries a class, and every goal
  decision a default and a `decide_by`. Nothing for me to arm this audit.
- **`D18`'s `decide_by` is TODAY (2026-09-09)** — armed, not yet overdue; its
  default fires tomorrow. `D24` 09-11, `D25` 09-13, `D19` 09-14, `D20` 09-18,
  and the new `D26` **09-10**.
- **Nothing acted on without being recorded.** `D17`'s firing (09-08) is
  correctly papered in `DECISIONS_NEEDED.md`, `DECISIONS_RESOLVED.md` and the
  84th audit's B1, with `GOAL.md` untouched and the premise-was-false caveat
  stated rather than buried.
- `UNROUTED-OWNER-ASK` 0/3, `VANISHED-OWNER-ASK` 0/0. `PROGRESS.md`'s one live
  owner ask (item 1) is correctly attributed to `D22` — which is RANK 3.

## SECTION 7 — bakeoff hygiene: CLEAN, with one disclosed wobble that stands

`PL.00/RENDER` (2026-09-07) seated **coarse-shadow512** at worst-seed 8.594
against an unmoved 5.0 floor. Two things about it, both already in the record
and neither retracted:

1. It was a **probe, not `run_bakeoff`** — no 3σ learning gate, because the arms
   are loop configurations rather than learners, so the gate has no referent.
   Stated on the entry. Correct.
2. **The winner is not the top scorer.** `coarse-flat` read 11.483 and lost on a
   "least-information-discarded" ranking **declared at `b7324ba` — the same
   commit that carries the artifact.** The 84th audit caught exactly this and
   wrote the lesson (*probe-class bakeoffs pre-register in a PRIOR commit*), and
   back-filled the record from an adjective to a pointer. The instance was not
   retracted and does not need to be — both arms clear the floor, so the gate is
   satisfied either way and the choice between them is a disclosed design
   preference. **No VOID treated as a verdict; no winner inside a noise margin.**

## SECTION 8 — the honest summary

**No. We are not closer to a curious humanoid that climbs the ladder than we
were yesterday, and today we are not even closer to a longer list of green
ticks.**

The demonstrated count has been 108/245 for two days. The last three things
written to the ledger were the ladder certifying its own measuring instruments.
The organ that could change that has been switched off for 22 hours by a gate
reading a meter that is 62% somebody else's, on a model meter that is 72%
somebody else's, and the week's entire free GPU allocation — 30 hours, the only
compute this project is permitted — will expire on Sunday with a good chance
that not one hour of it was used.

**What is genuinely good, and it is not small.** The ledger is sound: 108 PASS
rows, every commit alive, every control declared, zero stale PASS. Seven days of
spec and test diffs contain exactly one moved constant and it moved in the
strengthening direction. The one loosening-shaped edit is definitional and
disclosed. Two organs independently diagnosed the outage within ten minutes of
each other. And on 09-07 the system did the hardest thing it knows how to do:
`PL.00`'s edge was dissolved **by being satisfied rather than edited**, in the
week that edge produced an inconvenient FAIL.

**But four of the owner's constitutional commitments have no living claim at
all** — smell, balance, shelter, thermal — and *"too cold kills him"* has been
claim-dead since 08-25. One brain in unison is 1 of 27. Fast/slow is 0 of 8. The
instruments are excellent and getting better every day; what they measure has
not moved since Monday.

**The project is not in integrity trouble. It is in motion trouble.** The
honest verdict is DRIFTING, and the drift is not toward a wrong goal — it is
toward spending our best hours making the ruler straighter while the thing it
measures stands still.

---

## FOR THE BUILDER

**B1 — Fire `D22`'s default and write the record.** The owner did not rule by
2026-09-08. The OVERDUE NOTICE is appended to `DECISIONS_NEEDED.md` (RANK 3);
the resolution record in `DECISIONS_RESOLVED.md` is yours, because `D13` says
the overseer may not write it. Required wording, verbatim: *"the owner did not
rule by 2026-09-08, so the pre-registered default fired"*. What fires is **(i)
THE RULE STANDS** — design authority stays with the Review, unchanged and
unnarrowed; nothing is written, no threshold moves, `GOAL.md` is not touched.
State the reversal (the owner may rule (ii) or (iii) later at no cost) and state
that the deadline was **not** extended. Then `decisions --check` should print
`D22` off the overdue list.

**B2 — Count the dark slots. Instrumentation only; gate nothing.** This is
`D26`'s default (iv) *detection half*, and it is takeable now because it moves
no threshold, refuses no run and relaxes nothing — the same shape as the
`kindless_arena_discharges` and `AGEING-IN` precedents. Two numbers:
`builder_consecutive_skips` (consecutive `PACING:`/`REFUSING` lines in
`ladder.log` since the last real `iteration end`) and `builder_hours_since_pass`
(wall time since `ledger.json` last gained a PASS at a spec that is not Tier 0).
Both ratcheted, both printed in `status`'s RATCHET COUNTERS block. **Do not wait
for `D26`'s 09-10 default** — under (iv) it fires after the builder is forecast
to wake, and a counter commissioned by an outage should exist before the next
one. Add the known-positive to `T0.xx` in the same commit. If `D26` is later
ruled (i), this counter is what will prove the fix worked.

**B3 — Re-title `five-commitments-are-claim-dead-behind-foreclosures`.**
`coverage` now reads **4**, not 5. A queue row that names a count and silently
disagrees with the instrument it exists to discharge teaches its reader to trust
the title over the tool. Re-title with today's number and a one-line note saying
which commitment left the class and why. Do not touch the `DUE: 2026-09-11`.

**B4 — Print the sustained rate beside the record one, and do not move the
ratchet.** `MEASURED_DISCHARGE_CAPACITY = 6` is a one-cycle maximum;
`throughput()` measures 0.43 disposed/cycle sustained, 14× lower. The AMBER pile
warning divides by the maximum, so `review_queue_piled_on` fell 26 → 6 and
flagged dates 6 → 2 on the strength of one record-setting morning. Add a second
**reading** — dates over the *sustained* rate — printed beside the amber, as a
METRIC and never a violation. **Leave `MEASURED_DISCHARGE_CAPACITY` and the
`review_queue_piled_on` floor exactly where they are**; this adds a truer number
beside an optimistic one, it does not re-litigate the raise, which was
evidence-backed and honestly caveated in its own docstring.

**B5 — Standing, and it costs nothing: when the pace gate lets you back in
(forecast 2026-09-10T17:23), the first act of the first slot is a GPU
dispatch decision, not housekeeping.** W37 stands at **0.00 of 30** free
Kaggle hours with the quota expiring 09-13. Everything on the Review's dated
block that does not need GPU can wait an hour; 30 free hours cannot wait three
days. If the twin-spread probe for `D1.0` is ready, it goes first — forward
passes only, both branches pre-registered on the row, W37 and never W36.

---

## FOR THE OWNER

**1. `D22` — your deadline passed and the default fired. Nothing changed, and
that is the point.** You did not rule by 2026-09-08 on whether the builder may
draft redesigns, so **(i) THE RULE STANDS** fired: design authority stays with
the Review. Nothing was written, no threshold moved, `GOAL.md` is untouched. You
may still rule (ii) or (iii) at any later date at no cost. Worth knowing: the
Review, which made the ask, told you on 09-08 that its own evidence had weakened
and asked you to **wait a week and re-measure on 09-15** instead of granting it.
The default and the author's current preference agree.

**2. `D26` needs you by tomorrow, and I have added evidence that changes what
option (i) costs.** The Review is right that `pace_gate` starved the builder on
a shared meter — I confirmed every number independently. What it did not
measure: **`week:Fable` reads 95%, the loop's `MODEL_FLOOR` is 95, and 72% of
that meter is not the builder's either** (27 of 95 points are ours, summed over
28 measured runs; 48 points arrived in the 22 hours since the builder last ran).
So option (i) does not put the builder back on Fable — it puts it on **Opus**,
via `FALLBACK_MODELS`, billed against the very meter (i) has just stopped
gating. **I still think (i) is probably the right call**; the pace line should
measure what its own comment says it is about. But you should rule on it knowing
it buys Opus slots, not Fable slots. Appended to `D26` as an evidence addendum.

**3. The bill for this, so it is a number and not a worry: 30 free GPU-hours,
expiring Sunday, 0.00 spent.** This is the third time in six weeks the loop has
gone dark across a Kaggle expiry — 8.82 hours died in W32, 22.11 in W33, and
this week the whole 30 are exposed. Nothing in the project is broken; the
compute simply cannot be spent by an agent that is not awake. **If you want
those hours used, the two levers are yours and both are on your desk today**:
rule `D26` (i), or run `claude` less on this account between now and Sunday.
There is no third lever the system can pull for itself, and I would rather say
that plainly than route it as work.

**4. `D18`'s `decide_by` is today.** Its default is measure-and-report,
gate-nothing, relax-nothing — the SYSTEM.md ~1.5 GB memory ceiling stands
verbatim and is left visibly breached rather than quietly adjusted. It fires
tomorrow if you are silent, and firing it costs nothing and reverses in two
reverted commits. Also live: `D26` 09-10, `D24` 09-11, `D25` 09-13, `D19`
09-14, `D20` 09-18.

**5. NO-DECISION, for your awareness only: four of your own constitutional
commitments have no living falsifiable claim.** *"Too cold kills him"*
(thermal), smell, balance, and shelter-building — your own image of success —
are all CLAIM-DEAD: every claim spec parked or foreclosed, each parking legal
and evidence-backed at the time. Three of the four trace to a single cause the
Review named yesterday as one disease across five fronts: **our tasks are too
easy for our instruments to say anything about them** — nulls that hold the
roof, worlds that separate nothing, oracles that cannot. It is routed
(`five-commitments-are-claim-dead-behind-foreclosures`, DUE 09-11) and going to
the Sunday FULL as one question rather than five. Nothing for you to rule on;
I am telling you because it is the gap between the ladder and your goal, and it
has been open since 08-25.
