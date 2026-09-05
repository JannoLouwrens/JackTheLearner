# OVERSIGHT — 72nd audit, 2026-09-05 00:37–01:1x UTC (at `0861289`)

## VERDICT: DRIFTING — the ledger is clean and three of Jack's own claims have been sitting FAILED for a fortnight with no owner, no clock and no queue row. `XL.01` — *"death does not erase what he learned"* — has read FAIL for **17 days**, and every instrument this project owns reports it as fine.

Say the clean part first, because it is large, it was checked mechanically, and
it is true.

**Section 1 is clean.** All **104** PASS rows resolved: every one has an
implementation reachable through `run._module_for` (zero missing, including the
`SPEC_ID` indirections); every `commit` field resolves in git — **zero
dangling**; every one of the 104 declares a `control`; 102 of 104 wire it in
source, and the two that do not (`T0.01`, `T0.10`) declare
`"NONE, BY DECISION (52nd audit B5)"` in the registry, so the absence is
pre-registered rather than silent.

**Section 2 is clean — the fifth consecutive week.** Over the trailing 7 days
I diffed every commit touching `registry.py`, `registry_expansion.py` and
`experiments/tests/` and extracted every `CONST = number` that changed value.
**Seventeen constants moved and sixteen moved in the strengthening direction**
(`N_PROPERTIES` +1 five times, `N_LIVES` 16→32, `N_EVAL` 48→120,
`N_DECISIONS` 3200→4800, `LIVES_PER_ARM` 4→16→48, `COORD_MIN` 0.55→0.70,
`COORD_MARGIN` 0.20→0.35, `TEMP` 0.25→1.0, `STEPS` 300→500). **One moved
downward: `DECAY_MIN` 1.5 → 1.25** (`44f24c4`, 2026-08-29, T2.09). I read it
in full: it is a RIG bar on a spec whose `run()` had refused until that commit,
it is a placeholder being frozen from a pilot for the first time, seed 90's
claim arm read 1.472 so the placeholder would have discarded a live decaying
signal as dead, and 1.25 is derived from the gate's fixed point (a constant
signal decays by exactly 1.0) rather than shaved to the observed minimum. The
commit message says all of that in the open under the heading *"ONE BAR MOVED,
DOWNWARD, IN THE OPEN"*, and seven sibling bars were confirmed unmoved in the
same freeze. **Legal, measured, declared. No finding in section 2.**

**The SO.07 run in flight is real, verified rather than believed.** Worker pid
`1341501`, 35:57 elapsed against 35:51 of CPU (99.7%), RSS 252 MB, both pids in
`declared_pids`, launched through `launch_detached.sh` at 00:08:24, and
`/data/so07_hand_logs_s0.json` (11,159 bytes) was written at 00:25 — seed 0 is
done. The 201-byte log is the block-buffering the builder pre-declared, not a
hang. The dead-pilot scar cost 45 minutes last night and was paid; this launch
is clean.

Now the findings, ranked by how much they damage the ledger's trustworthiness.

---

## 1. THREE SETTLED FAILS HAVE NO REPAIR OWNER, NO QUEUE ROW AND NO CLOCK — 15 TO 17 DAYS OLD — AND FIVE INSTRUMENTS EACH REPORT THEM AS SOMEONE ELSE'S PROBLEM

This is the coverage miss of 2026-08-10 with a different mask on. That one was
about specs that did not exist. This one is about specs that ran, lost, and
then fell through every net at once.

**The measurement.** 22 specs carry a settled `FAIL`. For each I asked two
questions: does it declare a `repaired_by` (the reporting edge the 69th audit
B3 shipped for exactly this), and does its id appear anywhere in
`docs/REVIEW_QUEUE.md`? **Eighteen of 22 are routed by one or both.** Four are
not, and one of those four (`T3.07`) is dispositioned in its own registry notes
by `D7`'s fired default. **That leaves three genuinely unowned:**

| spec | FAILED | age | `COVERS:` declares it | title |
|---|---|---|---|---|
| **`XL.01`** | 2026-08-19 | **17 d** | **claim** — death & retry, memory across lives | *Death does not erase what he learned* |
| `T2.05` | 2026-08-20 | 16 d | fixture — fast/slow | *World model beats constant prediction* |
| `T4.02` | 2026-08-21 | 15 d | rule — one brain / unison | *No modality collapse* |

`repaired_by == []` for all three. Zero mentions across `REVIEW_QUEUE.md`.
`XL.01` appears in no open decision — its only occurrence in
`DECISIONS_NEEDED.md` is a stale table at line 1654 listing it under
*"specs ready to run"*, which it has not been since 08-19.

**Two of the three are GOAL.md's own sentences.** `XL.01` carries *"He lives,
he dies, he remembers… Life N+1 must be measurably better than life N because
of what life N recorded. Death is not a reset; it is a page turn."* `T4.02` is
the ladder's stage 4 verbatim: *"Unison (Tier 4): senses fused; each proven
load-bearing; **no modality collapse**."*

**Why no instrument says a word — and this is the part that generalises.** A
settled FAIL with an empty `repaired_by` and no queue row is invisible to every
reader this project owns, and each one is behaving correctly:

- `coverage`'s CLAIM-DEAD test asks *parked or foreclosed*. A FAIL is neither,
  so `death & retry` prints **`4 specs 0 pass 1 now`** with `XL.01 RUNNABLE`
  beside it. The commitment reads as covered-and-live. It is neither.
- `run blocked` ranks by transitive block mass. `XL.01` blocks nothing, so it
  never appears; `T2.05` and `T4.02` likewise.
- `coverage`'s QUEUE DEPTH excludes it explicitly — *"22 settled"* — because
  a settled spec is not a dispatch.
- `review-queue` can only age rows that exist. These were never routed, so its
  `OVERDUE`/`STALE` classes have nothing to fire on and it prints
  **`0 violations`** while three claims rot.
- `champions --check` sees `T3.07` only as *"declares no COVERS kind"*.

**The diagnosis is worse than neglect, and I checked it before writing it.**
`XL.01`'s FAIL *was* diagnosed: `INTEGRATION_QUEUE.md:458` records it as
power-blocked — *"cannot resolve 2× at 3 seeds × 8 lives"* — and carries that
finding forward into `NE.08`'s notes as a binding pre-run power calculation.
So the repair exists on paper. **`NE.08` is `blocked<-NE.01`, and `NE.01` is
itself a settled FAIL that `run blocked` ranks second in the project (frees 8,
impl unchanged 11 d).** The repair path for *"death does not erase what he
learned"* runs through two FAILs and is unreachable today — and not one file
in this repository says so in a form any tool can read.

**Damage.** Not to the certificates: nothing here is a false PASS, and I want
that stated plainly. The damage is to the *map*. `coverage` is this project's
highest-priority instrument by my own standing orders, and for three
commitments it is currently printing a green-shaped reading over a hole. That
is the precise failure mode the 64th audit already wrote up under *"a standing
red is a hiding place"* — except here the tool is not even red.

**The repair is a ratchet, and the ratchet must count the class, not the tidy
example** (`T0.31`'s P4/P5/P6 precedent). See FOR THE BUILDER B1.

---

## 2. `D21`'s DEFAULT FIRES ON THE SAME DAY AS THE EVENT IT COMMANDS, AND THE EVENT IS AT 06:37. THE WINDOW IS SIX HOURS AND NO INSTRUMENT CAN SEE IT.

The 70th audit found that `D21`'s `decide_by` fell five days *after* the event
its default commands, shipped `DEFAULT-ACTION-EXPIRED` to make that a printed
number, and moved the clock 09-11 → 09-05. **It priced the day. It did not
price the hour.**

- `D21` `decide_by` = **2026-09-05** (today).
- `decisions.py` marks overdue at `(today - decide_by).days > 0`, so the
  **earliest firing day is 2026-09-06**.
- `D21`'s default commands *"the **2026-09-06** FULL Review takes the W1 design
  as the FIRST DESIGN item on its docket"*.
- `crontab`: `37 6 * * * scripts/review.sh`. **The Review sits at 06:37.**

So the default's entire execution window is **00:00–06:37 on 2026-09-06** — six
builder slots (`7 * * * *`), and after 06:37 the commanded event is in the past
and the default is dead in the same way `D22`'s is, without any tool saying so.
`decisions.py`'s class is date-granular *by construction* — its docstring is
explicit that the comparison is `<=` against `decide_by` and that the
arithmetic *"reads one date against another"*. A same-day-but-too-late firing
is outside what it can express.

**What it costs if the window is missed.** `D21` is the entry that blocks the
**4 CLAIM-DEAD commitments** (smell, balance, shelter/building, thermal), the
**3 PARK-ON-AN-UNREACHABLE-RELEASE pairs**, and the World seat's undeclared
verdict. Nothing would go red. The next audit would find it as history.

**The mitigating fact, stated because it is real:** the builder has already
written the plan into its journal three times — *"the D21/D15/D16 firing is
tomorrow's 00:xx–05:xx slot"* — and `D15` and `D16` share the same `decide_by`.
So the risk is not that nobody knows; it is that nothing enforces it, and the
09-06 00:xx and 01:3x slots are already committed to SO.07's harvest.

---

## 3. `D22` — THIS ORGAN'S OWN ENTRY, ARMED YESTERDAY — TRIPS THE CLASS THE SAME AUDIT SHIPPED, AND IS STILL UNREPAIRED A DAY LATER

`decisions.py --check` prints, live:

```
[DEFAULT-ACTION-EXPIRED] D22
   the default names 2026-09-05, 2026-09-08 but decide_by is 2026-09-08 and
   the earliest firing is 2026-09-09 — on the day this fires, that action is
   in the past.
```

`D22` was armed by the **70th audit** (`726fed8`, 2026-09-04 12:51) — the same
commit series whose B1 shipped `DEFAULT-ACTION-EXPIRED`. The 71st audit
(18:40) did not repair it. It is now ~12 hours older.

**Being fair to the instrument and to the entry, because this one is not a
violation and inflating it would be the failure I am here to catch.** Both
dates in `D22`'s default are *provenance*, not commands — *"a pacing decision
(`D15`) that fires on 2026-09-05"* and *"silence through 2026-09-08 costs
approximately 17 further net queue rows"*. `decisions.py`'s own docstring
predicts this positive **by name and by example**, using `D15`'s date as the
illustration, and says the class is deliberately RATCHETED rather than blocking
because the exact narrowing — ATTRIBUTION, *whose clock is this date* — is
unwritten on purpose: *"tuning the regex until today's corpus reads zero is
fitting the instrument to the sample."* The ratchet reads **1/1 — at floor, ok.**

So the finding is not "a broken default". It is that **the only two repairs the
tool offers are both unavailable**: shortening `decide_by` cannot help (any
`decide_by ≥ today` puts 2026-09-05 in the past of every firing), and the
attribution syntax does not exist. An entry that trips a class it cannot clear
will sit at the floor forever and quietly convert a ratchet into wallpaper.
The cheap durable fix is B2.

---

## 4. TWENTY-FOUR ITERATIONS, ZERO SKIPS, +2 PASS — AND THIS PROJECT'S OWN COVERAGE TOOL CREDITS NEITHER PASS AS A CLAIM ABOUT JACK

Section 4, measured from `ladder.log` over 2026-09-04 00:40 → 2026-09-05 00:40:

| | value |
|---|---|
| iterations started | **24** (hourly, no gaps) |
| ended `rc=0` | **24 / 24** |
| `PACING:` skips | **0** (last skip 08-29) |
| demonstrated | 102 → **104** (+2) |
| registered | 234 → **242** (+8) |
| pass rate | 43.6% → **43.0%** (−0.6 pts) |
| ledger settlements | **12**, of which **3 first-ever** (`LG.03` VOID, `SO.06`, `SO.09`) |
| re-buys of existing certificates | **9 of 12** |
| commits | **66** |

The two PASSes are `SO.06` and `SO.09`. `coverage` classifies them from their
own `COVERS:` lines as **`SO.06 (fixture)`** and **`SO.09 (rule)`**, and prints
both under *"support passing, **not credited**"*. So the demonstrated count
moved +2 and the count of passing **claims about Jack** moved **+0**.

**I am not calling that drift, and here is why.** `SO.06` is the fixture that
certifies the owner's hands have a channel reaching only through the world;
`SO.09` is the accountant that makes puppeteering mechanically detectable; and
`SO.07` — *"what the hands leave is FOUND, and what he learns outlives them"* —
is the claim, and it is computing on this box right now. Building the guard
before the claim is the SO.06/LG.02 discipline working exactly as written, and
committing the implementation before the run is the discipline this project
paid for twice. **That is correct sequencing, not green-tick farming.**

What it does mean is that the honest read of yesterday is: **9 of 12
settlements were re-proving things already known, 3 were new, and the one new
thing that is a claim about the creature is still in flight.** The `+8`
registry growth against `+2` demonstrated is the runner falling behind the
ladder again after one day of the reverse.

---

## 5. COMPUTE HONESTY — NO FINDING, WITH NUMBERS

**GPU.** `W35` has **19.20 h charged of 30** (18.93 kaggle + 0.27 colab, 12
jobs); **~10.8 h free, expiring 2026-09-06 00:00**, i.e. in ~23 h. `16.18 h`
of that week is `D1.0`'s three jobs, returned VOID — GPU hours with no
certificate, correctly diagnosed by the 71st audit as a gate-design problem
whose repair is the two `d10-*` rows, not a re-dispatch. `W36` opens
2026-09-06 00:00 with 30 h and a named buyer. **The 09-04 Review's item 5 —
let W35's hours expire — is the right call and I endorse it: every runnable GPU
spec is a settled FAIL whose re-run is a seed lottery, or parked.** `overruns`
is empty.

**CPU.** Day 2026-09-05 has billed **1,800.0 s of 57,600** (all `SO.07`,
progressive billing by the outermost wrapper, matching the worker's 1,795 s of
CPU time to within a heartbeat — the meter is telling the truth). Yesterday
closed at 8,195.96 s. `cpu_foreclosed_now` moved 41 → 0 at the midnight
rollover; the builder recorded it and named it clock-coupled. **I checked the
counter's definition rather than the claim: `run.py:826` declares it
`"A METRIC with no floor"`, so the reset cannot manufacture a red and cannot
be used to re-base one.** Correct handling.

---

## 6. THE FOUR RATCHETS

| tool | rc | reading |
|---|---|---|
| `coverage` | **2** | 4 CLAIM-DEAD · 3 empty cost classes (`cpu<1min`, `cpu<48h`, `gpu<20min`) · 7 CITED-BUT-UNRUNNABLE (4 new: `GEN.02/03/06/09`, all `welded<-LC.07`, all routed on `goal-cites-four-specs-that-resolve-to-corpses` DUE 09-10) · unreachable **94 of 242**, baseline 94 — **AT floor, not above** |
| `decisions` | 0 | 0/10 undeclared · 1/1 DEFAULT-ACTION-EXPIRED (finding 3) · 1/3 UNROUTED-OWNER-ASK · 0 MEANS-ESCALATED |
| `champions` | 0 | 0 phantom arenas · 3/3 unfalsifiable · 3+1/4 uncontestable · 2/2 unverified verdicts · 3/3 trigger debt |
| `review-queue` | 0 | 0 violations · **34 live rows** · arrivals **4.43/cycle** vs disposals **0.14/cycle** · drain **UNBOUNDED** |

**`UNROUTED-OWNER-ASK: PROGRESS #4` cannot be cleared by this organ and I want
that on the record rather than carried silently.** The item is the Review's
organ-liveness paragraph — a report, not an ask — so the correct disposition is
`NO-DECISION:`, and `decisions.py` reads that marker **off `PROGRESS.md`
itself** (`owner_asks(progress_text)` sets `exempt`; the regex is `_NO_DECISION`
at line 482). `PROGRESS.md` is the Review's file; I may not write it. Routing a
liveness report into `DECISIONS_NEEDED.md` to make a number go down would be
paperwork about paperwork. **The Review owns this one line.**

**The 09-06 pile is now seven items against a measured capacity of one.** Six
live queue rows name 2026-09-06 (`w0-too-shallow` at **12 days**, both `d10-*`
gate rows, `lc07-checkpoint-branch`, `lt01-c2-body-cannot-rise`,
`cross-organ-doc-race-voids-certificates`), and `D21`'s default adds the W1
design as a seventh. One row due **today**: `t027-preserved-failimpl-as-artifact`.

---

## 7. SECTIONS 6 AND 7 — NO FINDING

`DECISIONS_NEEDED.md`: 8 open, all armed, **zero `MEANS-ESCALATED`** — no fork
a measurement could settle is sitting on the owner's desk. Nothing is armable
this audit (`0/10 undeclared`), so the arm-one-per-audit duty has nothing to
take; findings 2 and 3 are the clock work in its place.

`DECISIONS_RESOLVED.md`: 15 entries. I re-read the four most recent. The
2026-09-04 `run blocked`/`repaired_by` entry is the model of the class — it
records the losers, states the reversal in one line (`revert 9e847cf`), and
rests its authorisation on a verifiable invariant (*"`repaired_by` is read by
`cmd_blocked` alone"*) which I confirmed against `run.py`. **No decision made
without a gate, no VOID treated as a verdict, no winner inside the noise
margin.** `D10`'s single-arm caveat is carried on the seat's face, which is the
honest form.

---

## 8. THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the ladder?

**Marginally, and less than the counter says.**

Yesterday bought two PASSes and both are scaffolding. The one thing in the last
24 hours that could turn out to be a claim about the creature — *what the
owner's hands leave is genuinely FOUND, and what he learns from it outlives the
hand* — is running as I write and will settle around 01:35. That is a real
question about a real world with a real puppeteering control, and it deserves
its answer.

Against that: **13 of the 25 constitutional commitments `coverage` tracks still
have no passing claim.** Four are CLAIM-DEAD (smell, balance, shelter, thermal). Nine have live
claim specs and nothing passing — touch, tool use, told world, proprioception,
death & retry, plasticity, sleep, hunger/thirst, fast/slow. `curiosity` reads
2 passing of 12 specs. `one brain / unison` reads **1 passing of 25**, and its
declared rule — *no modality collapse* — has been FAILING for fifteen days
with nobody assigned.

And the shape of the whole week is unchanged and it is not the builder's fault:
24 slots a day, 24 clean exits, nine of twelve settlements spent re-proving
certificates, and a design desk that received 31 rows and disposed 1. The
project is not short of hands or of honesty. **It is short of decisions about
the world Jack lives in**, and the queue where those decisions live is
diverging at ≈4.3 rows per cycle.

The thing I would most want fixed by tomorrow is not any of that. It is that
*"death does not erase what he learned"* — which is close to the centre of what
this project is for — lost seventeen days ago, and the system that exists to
notice such things reported it as runnable, unblocked, and fine.

---

## FOR THE BUILDER

**B1 (from finding 1 — the one that matters).** Add a `FAIL-UNOWNED` class to
`coverage.py`'s report and to `run ratchets`: a spec whose ledger row is
`FAIL`, whose `Spec.repaired_by` is empty, **and** whose id appears in no
`docs/REVIEW_QUEUE.md` row. Live reading today is **3** (`XL.01`, `T2.05`,
`T4.02`); `T3.07` is excluded because `D7`'s fired default dispositions it in
the registry notes — **and that exclusion must be by a readable marker, not by
a hardcoded id.** Ratchet it at 3, shrink-only, and count the class rather than
the example (`T0.31` P4/P5/P6 shape). State in the docstring, as the other
three tools do, that the ONLY legal repairs are (a) routing a `REVIEW_QUEUE`
row with a `DUE:`, (b) declaring `repaired_by`, or (c) an explicit registry
disposition — **never deleting the FAIL row, never re-running for a better
number, and never adding to a baseline.** Add the corresponding falsifier to
`T0.17`/`T0.28`'s property set so the guard is itself on the ladder.

**B2 (from findings 2 and 3 — the clock, both halves, one commit).**
*(a) `DEFAULT-ACTION-SAME-DAY`.* `decisions.py` should name a default whose
referenced date **equals its earliest firing day** (`decide_by + 1`) as a
distinct, non-blocking class, because on that day the action is a race rather
than a certainty. `D21` is the live positive: it must fire before
`crontab`'s `37 6 * * *`. The class needs no hour parsing — equality of dates
is enough, and the report line can simply say *"this must fire before the event
it commands, on the same day"*. *(b) ATTRIBUTION, the narrowing
`decisions.py`'s own docstring calls the durable one.* Give the default text a
marker — `CLOCK: D15` / `CLOCK: consequence` in the same idiom as `COVERS:` and
`DECIDE:` — so a date that is provenance can say so and stop reading as a
command. That lets `DEFAULT-ACTION-EXPIRED` shrink 1 → 0 by **declaration**
rather than by regex-tuning, which is the thing the docstring refuses to do.
Both halves are declaration-parsing, not heuristics.

**B3.** Fire `D21`, `D15` and `D16` in the **00:07 slot on 2026-09-06, before
anything else including the SO.07 harvest.** All three have `decide_by
2026-09-05`; `D21`'s commanded event is at 06:37 that same morning. Journal it
with the mandated words — *"the owner did not rule by 2026-09-05, so the
pre-registered default fired"* — and state the reversal for each. If the owner
rules today, none of this fires and the entries close normally.

**B4.** Route the three orphan FAILs the moment B1 has a number, using
`next_free_due` (**2026-09-13**), not Sunday. `XL.01` first and with its
diagnosis attached — the power calculation in `INTEGRATION_QUEUE.md:458`, and
the fact that its named successor `NE.08` is `blocked<-NE.01`, itself a FAIL.
The row's question is *"what buys death-and-retry a reachable repair path"*,
not *"re-run XL.01"*.

**B5. Standing prohibitions, all carried forward unchanged:** do not re-dispatch
`D1.0` (gate design owed at the Review 09-06); `HR.1`–`HR.4` stay `D19`-held to
09-14, no corpus fetch; `HR.6` stays behind `HR.5`; `LF.01` attempt 2 waits for
the 09-09 design and `FIXTURE_VOID_CAP=3` is not permission; **no third
increment of the CPU accountant** — and I re-read your own 09-04 journal note
distinguishing *"removed a throttle"* from *"added a meter"* and accept it, but
the Review's prohibition stands for anything further; **let `W35`'s ~10.8 h
expire at 09-06 00:00** — inventory, not uptime; do not re-stagger the 09-06
docket a third time.

---

## FOR THE OWNER

**1. Three of Jack's claims lost a fortnight ago and nothing in this system is
assigned to them — including one that is close to the centre of what you asked
for.** `XL.01` — *"Death does not erase what he learned"*, which its own
`COVERS:` line files under both **death & retry** and **memory across lives** —
has read FAIL since **2026-08-19**. `T4.02` — *"No modality collapse"*, stage 4
of the path in GOAL.md, verbatim — since **08-21**. `T2.05` — the world-model
fixture behind fast/slow — since **08-20**. None has a repair owner, a queue
row or a deadline. Eighteen of the other nineteen settled FAILs have at least
one. **Nothing was hidden and nobody lied**: the certificates are honest, the
FAILs are recorded, and `XL.01`'s cause was even diagnosed (it lacks the
statistical power to resolve a 2× effect at 3 seeds × 8 lives, and that finding
was carried into `NE.08`'s notes). The defect is that the diagnosis went into
prose, `NE.08` is blocked behind `NE.01` — itself a FAIL — and **no instrument
can read a repair path that runs through two failures.** So `coverage` prints
`death & retry … 1 now` and the map shows solid ground. I have ordered the
counter (B1) and the routing (B4); this needs no ruling from you, and I am
telling you because it is the second time in four weeks that a whole class of
problem turned out to be invisible rather than unattended, and that pattern is
worth your eye even when each instance gets fixed.

**2. `D22` is on your desk and its own clock cannot be repaired — please read it
before 2026-09-08.** `D22` carries the Review's largest strategic
recommendation (*let the builder DRAFT redesigns; keep ratification at the
Review*), and today's numbers are the same numbers, unmoved: **34 live queue
rows, 4.43 arriving per cycle against 0.14 disposed, drain UNBOUNDED, six rows
promised to this Sunday against a measured capacity of one, and every one of
the startable specs behind that desk.** Its pre-registered default is *(i) the
rule stands* — the status quo, which by the entry's own stated price costs
about 17 further net rows through 09-08. Separately, `D22` trips
`DEFAULT-ACTION-EXPIRED`, which this organ armed and this organ shipped on the
same day, and neither of the two repairs the tool offers can clear it (the
dates are provenance, and the attribution syntax does not exist yet — B2 builds
it). I am **not** recommending a ruling. I am saying the entry is the biggest
thing on your desk and the deadline is Tuesday.

**3. Three defaults fire tomorrow morning on your silence: `D15`, `D16`, `D21`
(all `decide_by 2026-09-05`).** `D21`'s is the load-bearing one and it is
narrow by design — it re-orders one Sunday docket, touches no GOAL.md text,
moves no threshold, and reverses by re-ordering back. It has a **six-hour
firing window** (00:00–06:37 on 09-06, because the Review it commands sits at
06:37) and no instrument that would notice a miss; B2 and B3 close both halves.
`D15` adds a pace check to this organ's own cadence plus usage attribution;
`D16` keeps `T0.27` red rather than green. All three are reversible by revert.
**If you would rather rule than let any of them fire, today is the day.**

**4. Organ liveness — all green, verified against `/data/jack-logs` mtimes and
`ps`, not against anyone's report.** Builder 00:12 (hourly; 24 starts in 24 h,
24 `rc=0`, 0 `PACING:` skips), overseer 00:37 (this run, 6-hourly), Review 09-04
06:48 (daily 06:37, next fire in ~6 h — Sunday 09-06 is the FULL sitting), field
watch 08-31 05:53 (Mondays, next 09-07, inside cadence). `lost_iterations.log`
is still 0 bytes and still never exercised. `SO.07`'s registered run is 36
minutes into ~85 and genuinely computing. No organ is silent.
