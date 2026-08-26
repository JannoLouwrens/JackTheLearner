# OVERSIGHT — 33rd audit, 2026-08-26 06:37 UTC

## VERDICT: DRIFTING — but **not for the reason the last three audits gave.** The burn that starved the builder stopped five hours ago, the pace deficit is now *closing*, and the 32nd audit's owner escalation ("cannot close by itself", "29.69 GPU-hours expire") is **falsified by measurement**. What is actually broken is *this organ*: three audits in eighteen hours, three near-identical DRIFTING reports, ~3 points of the very meter that gates the builder, and **zero of their repairs executed** — because the only organ that can execute a `FOR THE BUILDER` item is the one the gate locked out.

Nothing in the repository has changed since the 32nd audit committed at 00:48.
`HEAD` is still `4e763b8`; the working tree is still the single untracked
`sm_03_nose_reports_occluded.py`; `84/187 demonstrated` is unchanged; every
hourly slot since has been pace-skipped. **An audit of a frozen system
re-measures a frozen system** — so this report deliberately spends its length on
the two things that *are* new, and states the carried-over findings by reference
rather than re-deriving them.

**Clean results, re-run by me rather than relayed:**

- **§1 ledger integrity — clean.** I ran `run verify` myself: **83 PASS entries
  re-judged from the record alone, 81 controls probed.** 0 verdicts that no
  longer re-derive, 0 gates that ignore their control, 0 controls declared but
  never run, 0 gates unreplayable, 0 entries unauditable, 0 controls run but
  undeclared. Two PASSes carry no control at all (`T0.01`, `T0.10`) — both
  existence claims, both long-declared, unchanged. `T0.18` self-excludes
  correctly.
- **§2 thresholds and controls — clean**, with one honest nuance the 32nd audit
  got slightly wrong. See §2.
- **§5 compute accounting — clean.** `overruns: []`; W34 charged **0.3111 h of
  30**, one job, one real ledger row (`T2.15` FAIL, fully written up).
- **§7 bakeoff hygiene — unchanged and clean.** No new bakeoff has run.
- The three constitutional gates all exit 0, all run by me:
  **coverage** — 0 commitments with no declared spec, 0 CLAIM-DEAD, 4 known
  dangling GOAL.md citations (shrink-only baseline);
  **decisions** — ratchet ok (3/10 undeclared, all three false — verified
  independently, see §6);
  **champions** — ratchet ok (6/8 seats with a phantom arena).

---

## RANK 1 — the 32nd audit escalated an emergency to the owner that **measurement has since falsified**, and the escalation channel is the thing that gets damaged

The 32nd audit told the owner, at 00:37:

> *"`week:all models` is at **51%** and rising **1.33 pts/h**; the pace line is at
> **42%** and rises **0.387 pts/h**. The gap has widened from 0 to 9 points in
> eleven hours and **cannot close by itself.** The loop hits the hard 90% stop
> around **Aug 27 05:00 UTC**. … without an intervention, **29.69 free GPU-hours
> expire unused for the third week running**."*

**Five hours of measurement since, from `ladder.log` — the same instrument:**

| slot (UTC) | 01:07 | 02:07 | 03:07 | 04:07 | 05:07 | 06:07 |
|---|---|---|---|---|---|---|
| `week:all models` (**the gate**) | 52 | 52 | 52 | 52 | 52 | **52** |
| week elapsed | 26 | 27 | 27 | 28 | 29 | **29** |
| pace line `25 + ⌈65·e/100⌉` | 42 | 43 | 43 | 44 | 44 | **44** |
| **gap over the line** | **10** | 9 | 9 | 8 | 8 | **8** |
| `week:Fable` (printed, not the gate) | 86 | 86 | 86 | 86 | 86 | **87** |

**The burn is 0.00 pts/h across five consecutive readings.** The gap has not
widened; it has **narrowed from 10 to 8**. Both of the 32nd audit's projections
are dead:

| projection (32nd audit, 00:37) | measured by 06:07 |
|---|---|
| `all models` rising 1.33 pts/h | **0.00 pts/h** for 5 h |
| gap "cannot close by itself" | **closed 2 points** unaided |
| 90% hard stop at ≈ Aug 27 05:00 | spend is **52%**, flat, not tracking to 90% |
| `week:Fable` 1.8 pts/h → 100% at ≈ Aug 26 08:00 | **87%**, +1 pt in 6 h (0.17 pts/h) → ~Aug 29 |

**When the builder actually wakes, computed from the gate's own arithmetic.**
`pace_gate` skips while `pct >= allow`, `allow = 25 + (65·elapsed + 99)/100`
(integer). With `pct = 52`, the loop proceeds at the first slot where
`allow >= 53`, i.e. `elapsed >= 42`. Elapsed is **29%** and rises **0.595 %/h**
(168-hour week). 13 points ÷ 0.595 = **21.8 hours → first eligible slot
≈ 2026-08-27 05:07 UTC**, with no intervention at all.

**And that is comfortably before the deadline the escalation was built on.**
Kaggle's `%U` week 34 (Sun Aug 23 – Sat Aug 29) expires **2026-08-30 00:00**.
From Aug 27 05:07 that is **~67 hourly slots**, for a `GPU_SHORT` dispatch
estimated at **0.4 h**. The 29.69 free GPU-hours are not lost on the current
trajectory; they were declared lost on an extrapolation.

**Being fair to the 32nd audit, because this matters.** Every *substantive*
finding it made is real and still unrepaired — the untracked `SM.03`, the dead
pilot, the harvest-path scope, the three false `UNDECLARED` entries, the
`week:Fable` blindness. Its attribution work (exonerating the `/usage` probes by
opening the transcripts rather than reasoning about them) was exactly right. What
failed was one specific move: **it took a rate measured over eleven hours of an
exogenous process it does not control, projected a date from it, and escalated
that date to the owner as an emergency requiring intervention.**

**Why this is RANK 1 rather than a footnote.** This project's most expensive
scar, `D1`, was twenty days of an owner escalation nobody could act on. The
countermeasure is that escalations must be *credible*. An escalation that says
"29.69 hours die unless you act" and is falsified by the next audit spends
credibility that the *next* escalation needs. And the pattern is now visible
across three audits:

- **31st audit** — measured 1.0 pts/h and **explicitly refused to extrapolate.**
  That was the correct behaviour and it should be the standard.
- **32nd audit** — measured 1.33 pts/h, extrapolated two dates, escalated. Both
  falsified within six hours.

This audit's own forward number above (`Aug 27 05:07`) is stated with its
condition attached and it is the load-bearing part: **it holds only while the
foreign sessions stay quiet.** Both are still live — `68804e98` (30.3 MB) last
wrote at **06:09**, `b76c8195` (5.9 MB) at **05:53** — so they are writing but
consuming almost nothing. If they resume at the 1.17 pts/h rate measured
13:07–01:07, the deficit re-widens and the Aug-30 risk returns. **The honest
output is a rate and a sensitivity, never a date.** Lesson appended to
`LESSONS.md` by this audit.

---

## RANK 2 — the Review and the overseer collide at **06:37 every single morning**, 30 minutes before the builder's slot — and nobody has reported it

New finding, and it is structural rather than transient. From `crontab -l`:

```
 7 * * * *  ladder_loop.sh      (builder — every hour at :07)
37 */6 * * *  overseer.sh        (fires 00:37, 06:37, 12:37, 18:37)
37 6 * * *  review.sh            (fires 06:37)
```

`*/6` includes hour 6. **Every day at 06:37 the two heaviest non-builder Opus
organs start in the same minute.** I am watching it happen as I write this:

```
1744222  05:08  timeout 20m claude -p REVIEW MODE TODAY: DAILY …
1744263  05:07  timeout 25m claude -p You are the OVERSEER of the JackTheLearner …
```

Two long-effort Opus sessions running concurrently on a **shared weekly meter**,
finishing at ~07:00 — and the builder's next slot is **07:07**, where
`pace_gate` reads that meter. The scheduler is arranged so that the single hour
most likely to be over the pace line is the one immediately after both auditors
just spent on it, every morning.

This is not the *cause* of the current 18-hour blackout (that was the foreign
sessions — the 32nd audit attributed it correctly). It is a standing defect that
makes the 07:07 slot the most likely to be lost, forever, and it costs nothing to
fix: move the overseer off the collision. `37 3,9,15,21 * * *` keeps the
6-hourly cadence, clears the Review entirely, and places each audit mid-way
between builder slots instead of immediately before one.

---

## RANK 3 — three consecutive audits, ~3 points of the gating meter, **zero repairs executed** — the oversight organ is now consuming the resource it is reporting on

| time (UTC) | organ | outcome | repairs executed |
|---|---|---|---|
| 08-25 12:46 | 30th audit | DRIFTING | 0 |
| 08-25 18:47 | 31st audit | DRIFTING | 0 |
| 08-26 00:48 | 32nd audit | DRIFTING | 0 |
| 08-26 06:37 | this audit | — | 0 possible |

The builder ran its last iteration at **08-25 12:23**. Every audit since has
written a `FOR THE BUILDER` section. **`pace_gate` runs at `:07`, before the
iteration ever opens this file** — so the loop cannot distinguish *"there is
nothing urgent"* from *"there is a flagged emergency and I am not permitted to
look at it."* The 32nd audit named this deadlock; what it did not do is draw the
conclusion about **its own cadence**.

**Measured cost of an audit against the gate.** Spend was 51% at 00:07 and 52%
at 01:07, spanning the 32nd audit (00:37–00:48) in a window with no builder
iteration and no other jackthelearner organ. **≈1 point of the weekly
`all models` meter per audit** — the same meter, and the same units, as the
8-point deficit currently keeping the builder dark. Four audits/day is ~4
points/day of the resource whose scarcity is the audit's own headline finding.

**The repair, and it is better than the cadence cut the 32nd audit offered the
owner.** A blanket move to `37 */12` halves oversight even when the system *is*
moving, which is exactly when oversight is worth most. A **change-gated no-op**
costs nothing when there is nothing to see and full coverage when there is:
`overseer.sh` skips a slot when *all* of —

1. `git rev-parse HEAD` is unchanged since the last audit, **and**
2. `ladder.log` records zero iteration starts since the last audit, **and**
3. no `decide_by` date in `DECISIONS_NEEDED.md` falls before the next slot, **and**
4. fewer than 3 consecutive slots have already been skipped (so a **full audit
   runs at least every 24 h regardless**)

— hold, logging `overseer: no-op, HEAD 4e763b8 unchanged and 0 builder
iterations since <ts>`. Conditions 3 and 4 are what keep it from being a
blindfold: a deadline firing or a detached run landing still gets an audit, and
the organ can never go dark for a day. Armed as **D13** in `DECISIONS_NEEDED.md`
by this audit, with the no-op as the pre-registered default on 2026-08-31 — it
strictly *reduces* spend and weakens no gate, so it is within already-permitted
actions.

---

## RANK 4 — `SM.03` has now been untracked for **18 hours**, and it is the only asset the expiring GPU quota could buy

Carried from the 31st audit (which *predicted* it) and the 32nd (which ordered
the repair). Unchanged, and therefore worse:

```
?? experiments/tests/sm_03_nose_reports_occluded.py    710 lines, untracked since 08-25 12:21
```

- `/data/sm03_pilot_seed90.json.log` — **0 bytes**, mtime 12:21. The result JSON
  was never created. pid 1552865 does not exist.
- The `12:07` iteration closed `rc=0` reporting *"pid 1552865, ~667 MB,
  healthy"* and *"I'll be re-invoked when it completes."* None of that was true
  14 minutes later.
- **A 0-byte log with a resident 667 MB process is the import-death signature**
  this box already has a lesson for: a `/data`-rooted detached script that never
  reaches its first write. The docstring's pilot numbers are **still owed** and
  its gates are **not frozen** — so it must not be dispatched until a pilot
  actually writes a result, whatever the GPU clock says.

`SM.03` is the successor claim spec for **smell**, registered two iterations ago
specifically to un-CLAIM-DEAD one of the owner's constitutional senses, and it is
the single best `GPU_SHORT` candidate for W34's 29.69 remaining hours. It is one
`git clean` from gone, and the one path that runs during a pace skip
(`HARVEST_PATHS`) is scoped to `ledger.json`, `gpu_budget.json`,
`gpu_submissions.jsonl` — verified in `ladder_loop.sh:125`. **The rescue path
covers the one artifact class that is not the problem.** Second consecutive
audit to say so.

---

## §2 — thresholds and controls, seven days: no loosening, and one correction to the last audit

I ran the scan independently (`git log -p --since="7 days ago"` over
`registry.py`, `registry_expansion.py`, `experiments/tests/`, plus `git log -S`
on every threshold symbol appearing in a removed line). **No threshold moved in
the loosening direction. No control was deleted or weakened. No `_check` gained
an `or`. No seed count was reduced. No assertion was removed.**

The two things worth naming rather than waving through:

- **The 32nd audit wrote *"every numeric change is an addition."* That is not
  quite right, and the exception deserves to be on the record rather than
  glossed.** `ddbe6b7` (NE.01) moved `DELTA_T_NIGHT` **12 → 10**. It is not a
  loosening: the commit carries the sweep that justifies it (occlusion 0.598, the
  *edge* of the declared 0.3–0.6 validity band → 0.498, mid-band), the move is a
  **rig calibration into the middle of a pre-declared band**, the *claim* gate is
  untouched, it happened **before** the run per pre-registration order, and
  `NE.01` FAILed anyway — so no credit was bought by it. Correct in substance;
  the previous audit's summary sentence was just too strong.
- The single deletion that a line-grep flags in `registry.py` is `DP.04`
  **gaining** `LG.00` in `depends_on` and being reflowed onto two lines — a
  dependency *added*, tightening the graph. Chased down, not trusted.

Guards observed firing in-tree: `## do not add seeds; 27th audit B1`, and
`protocol.py`'s `UndeclaredControl` refusing `T2.15`'s first dispatch at 0.0 s
with nothing spent.

## §3 — drift from the goal

**The builder did no work in the audit window** (last iteration 08-25 12:23), so
there is nothing new to trace. The four iterations before it all traced cleanly
to GOAL.md and the 32nd audit's table stands.

The converse question is the one that has not moved and is the real answer to §8:
of 24 constitutional commitments, **14 have live claim specs and nothing
passing** — `smell`, `voice`, `balance`, `thermal (kills)`, `shelter/building`,
`proprioception`, `plasticity`, `sleep`, `social`, `tool use`, `touch/contact`,
`fast/slow` (8 declared specs, 0 passing), among them. `curiosity` has 12 specs
and **1 PASS**; `one brain / unison` has 21 specs and **1 PASS**. These are the
three claims GOAL.md calls the thesis itself.

## §4 — is the builder alive and productive?

**Alive but blocked, and blocked correctly by its own rules.** Last iteration
start `2026-08-25T12:07`, end `12:23:33`, `rc=0`. Every slot since —
**eighteen consecutive hourly slots** — logged `PACING: … skipping`. No crash, no
credit exhaustion, no repeated identical failure, no paused loop nobody resumed.
The cron entry is intact and the gate is behaving exactly as designed.

Iterations in the last 24 h: **0**. PASS delta over the same window: **0**.
`84/187 demonstrated` (44.9%, down from 49.1% as the registry grew 169 → 187
while PASS stood still). Last PASS of any kind: `T0.21`, 08-25 10:14 — a *guard*
re-stamp. Last **claim**-kind PASS: `T3.01` (sight), **2026-08-21 01:28 — 5.2
days ago.**

## §5 — compute honesty

W34 charged **0.3111 h of 30**, a single job (`jack-ladder-1787631708`, `T2.15`),
which produced a real ledger row with a full write-up and a routed follow-up.
**No GPU hour this week was spent without a ledger entry.** `overruns: []`. The
`%U` (Sunday-start) week key remains correct by design and matches Kaggle's real
reset; the W32 opening-balance gap (6.3849 h) is still carried in the
over-stating direction, which is the safe one. **The accounting is sound; see
RANK 1 for why the *spending* forecast has changed.**

## §6 — stuck decisions

Eight open decisions, all armed, all `decide_by: 2026-08-31`, none overdue, none
`MEANS-ESCALATED`. No owner decision was acted on without being recorded.

I verified the three `UNDECLARED` entries myself rather than relaying the 32nd
audit's conclusion, because an audit declaring a ratchet alarm "false" is exactly
how a ratchet leaks. It is right: `DECISIONS_NEEDED.md:274` reads
`## ~~D3 — May the loop `git push`?~~ **ANSWERED: YES (owner, 2026-08-10)**`,
and the two title-keyed entries (`:408`, `:558`) carry owner rulings dated
2026-08-09 in their bodies under headers still reading `(OPEN, …)`. Note the
compounding defect already recorded at `:341` — `_DECIDE` forbids spaces in an
id, so **a title-keyed entry cannot be armed at all**; the settled-header repair
(B2) is the only route for those two.

**This audit arms one decision — `D13`, the overseer's own cadence.** It is not
a manufactured fork: the 32nd audit put exactly this question to the owner in
prose (*"cut me to `37 */12`"*) with **no default and no clock**, which is the
`D1` shape verbatim. Arming it is the standing duty. Its default (the
change-gated no-op of RANK 3) strictly reduces spend, edits nothing owner-owned,
and weakens no gate — a narrowing, as a default must be.

## §7 — bakeoff hygiene

No bakeoff has run since the last audit. Re-read rather than assumed: `PS.01/J`
stands as a **VOID**, not laundered into a verdict; `PS.01/J2` declares its
2.66σ margin and its `screen` rationale openly; `D2` was resolved by ledger
replay with its losing branch recorded. **No findings.**

## §8 — the honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
No — and today the reason is different from yesterday's, which is the finding.**

Yesterday's answer was "the builder is starving on somebody else's spend." That
was true and it is now **resolving on its own**: the burn stopped, the deficit is
closing, and the arithmetic says the loop wakes ~Aug 27 05:07 with ~67 slots and
29.69 free GPU-hours still in front of it. The emergency the last audit put on
the owner's desk did not need the owner.

What replaced it is smaller and more embarrassing. In the eighteen hours the
builder sat gated, this project ran **three Opus audits**, produced **three
near-identical DRIFTING reports**, spent **~3 points of the exact meter holding
the builder down**, executed **zero** of its own repairs, and let a 710-line spec
aimed straight at a constitutional sense sit **untracked** for a third
consecutive audit. The oversight layer stayed perfectly busy while the thing it
oversees was switched off. That is the "polishing the machine instead of running
it" failure `SYSTEM.md` names in its hard constraints, and the organ doing it is
this one.

The instruments remain genuinely good: the ledger re-verifies clean at 83
entries with 81 controls probed, not one threshold has moved the wrong way in
seven days, all three constitutional ratchets are green, and the champions
ratchet has shrunk 8 → 6. I believe those results. **But 84 PASS has not moved in
six days, the last capability PASS was 5.2 days ago, and 14 of 24 constitutional
commitments have a falsifiable claim registered and nothing behind it.** We are
measurably better at knowing what we have not proved. That is worth something —
it is not worth a week.

---

# FOR THE BUILDER

Ordered by damage. **B1 and B2 execute during a pace skip and must**; they are
the only repairs that can run before ~Aug 27 05:07.

**B1 — rescue the orphan, then widen the rescue path (third audit asking).**

1. Commit `experiments/tests/sm_03_nose_reports_occluded.py` (710 lines,
   untracked since 08-25 12:21). State in the commit message that its seed-90
   pilot **wrote zero bytes** — `/data/sm03_pilot_seed90.json.log` is 0 bytes,
   `/data/sm03_pilot_seed90.json` was never created — so the docstring's pilot
   numbers are **still owed** and its gates are **not frozen**. **Do not dispatch
   it until a pilot actually writes a result**, GPU clock notwithstanding.
2. In `ladder_loop.sh`, add a second, separately-committed rescue beside
   `harvest_bookkeeping()` for untracked/modified files under
   `experiments/tests/` and `docs/` — a mechanical *"orphaned work is versioned,
   never judged"* commit, reusing the existing torn-file guard and an explicit
   pathspec (the `add -A` ban stands). **Never** `experiments/ledger.json` by
   this path — only the runner writes verdicts.

**B2 — settle the three false `UNDECLARED` entries by document edit, not regex.**
Unchanged from the 32nd audit and still correct: do **not** add `ANSWER` to
`_SETTLED` — `DECISIONS_NEEDED.md:1454` reads
`## D1 — DO NOT ANSWER "DO WHAT THE MEASUREMENTS SAY"`, and `_SETTLED` closes a
key if *any* header matches, so that widening silently closes **D1, the 38-spec
decision**. Append one settled header per entry using the token the tool already
owns (`RESOLVED`), quoting the ruling already in the body. Ratchet shrinks
3 → 0, blast radius zero.

**B3 — move the overseer off the Review's cron slot (RANK 2, new).**
`37 */6` and `37 6 * * *` collide at **06:37 every day**, putting two concurrent
long-effort Opus sessions on the shared meter 30 minutes before the builder's
07:07 slot. Change `crontab` to `37 3,9,15,21 * * *`: same 6-hourly cadence, no
collision, each audit mid-way between builder slots. Update `scripts/crontab.txt`
in the same commit so the file and the live crontab do not diverge.

**B4 — the change-gated overseer no-op (RANK 3).** Implement the four-condition
skip specified in RANK 3 — `HEAD` unchanged **and** zero builder iterations since
the last audit **and** no `decide_by` before the next slot **and** fewer than 3
consecutive skips. Conditions 3 and 4 are load-bearing: without them this is a
blindfold rather than a saving. **This is `D13`'s pre-registered default and
fires 2026-08-31 if the owner does not rule** — implementing it early is fine,
reverting it is one commit.

**B5 — artifact check on every detached launch (second audit asking).** After
launching a detached pilot, wait ~10–15 s and assert its log is **non-empty**
before reporting the launch succeeded. A 0-byte log is an import death, and it
must be reported as a failure **in the same iteration**. **RSS is not liveness**
— the SM.03 pilot was 667 MB resident and wrote nothing. Third occurrence of this
shape in four audits.

**B6 — make the builder's own exhaustion visible.** `week:Fable` is at 87% while
both gates read `all models` at 52% and say proceed. Add a **pre-flight check,
not a new limit**: if the loop model's own weekly line is ≥ 95%, log
`ABORT: builder model <M> exhausted (<n>%) — the gate reads 'all models' (<m>%)`
and exit 0 without consuming the slot. It only ever refuses *more* than the 90%
stop, so it cannot weaken it. (Note: the 32nd audit's "Fable hits 100% at
08-26 08:00" was extrapolation and is falsified — measured 0.17 pts/h, ~Aug 29.
The blindness is still real; the urgency was not.)

**B7 — when the loop wakes (~Aug 27 05:07), the first unit is `SM.03`'s pilot,
not a new spec.** A real seed-90 pilot, gates frozen from its numbers, then
`dispatch.sh SM.03`. It is the only registered `GPU_SHORT` claim spec standing
between 29.69 free hours and a constitutional sense with zero passing claims.

**B8 — `decisions.py` silently overwrites a wrapped `default:`, and I hit it
arming `D13` this hour.** `_FIELD` matches any indented `word: value` line as a
**new key**, while the continuation rule only applies to indented lines that are
*not* key-shaped. So a `default:` whose prose wraps such that a continuation line
begins `default:`, `class:`, `blocks:` or `decide_by:` **replaces the field
rather than continuing it** — and the tool then reports the entry as armed, with
a fragment as its default. My first `D13` block armed the sentence *"halving the
cadence unconditionally cuts oversight hardest…"*, which is an argument *against*
the option I was defaulting to. I caught it only because I re-ran `--check` and
read the printed default instead of trusting the exit code.

This is the same family as the bug already fixed one layer down (*"the first
parser silently dropped these, which truncated every wrapped `default:` to its
first line — a default that reads as half a sentence is worse than none, because
it still looks armed"*). The repair then was continuation; the hole left is that
continuation loses to key-matching. Concretely: **refuse a duplicate key** —
if a `DECIDE:` block declares the same field twice, raise rather than
last-write-wins, since no legitimate entry does that and the failure is
otherwise invisible. Add a `_fixture()` case with a wrapped default whose second
line starts `default:`. **This matters more than it looks: an armed default that
says the opposite of what its author meant will FIRE on 2026-08-31.**

---

# FOR THE OWNER

**Please disregard the intervention my predecessor asked you for six hours ago.**
The 32nd audit told you the builder's pace deficit *"cannot close by itself"*,
that the loop would hit its hard stop around Aug 27, and that 29.69 free
GPU-hours would expire unless you acted. **Five hours of measurement since say
otherwise, on the same instrument:**

- `week:all models` has been **flat at 52% for five consecutive hourly readings**
  — a burn of **0.00 pts/h**, against the 1.33 pts/h that projection assumed.
- The gap over the pace line has **narrowed from 10 points to 8**, not widened.
- By the gate's own arithmetic the builder resumes on its own at
  **≈ 2026-08-27 05:07 UTC** — roughly **67 hourly slots** before Kaggle's free
  30 h expire on Aug 30, for a job estimated at 0.4 h.

**No action is needed from you on this.** The one condition worth knowing: both
long-running sessions in `/home/opc` are still open (last writes 06:09 and
05:53) but are currently consuming almost nothing. If heavy use resumes at the
earlier rate the deficit re-widens and the Aug-30 risk comes back — closing or
pausing them would remove that risk entirely, but on today's measurement it is a
convenience, not a rescue. **I am not asking you to touch the 90% ceiling and
would push back if it were proposed.**

**One decision, newly armed, and it is about me.** `D13` in
`DECISIONS_NEEDED.md`: in the eighteen hours your builder sat gated, this
overseer ran three times, wrote three near-identical reports, spent roughly one
point of your weekly meter per run — the same meter and the same units as the
8-point deficit keeping the builder down — and executed none of its own repairs,
because the only organ that *can* execute them was the one locked out.

The default, if you do not rule by **2026-08-31**, is that `overseer.sh` learns
to skip a slot when nothing has changed: same `HEAD`, zero builder iterations
since the last audit, no decision deadline due before the next slot, and never
more than 3 skips in a row — so a full audit still runs at least daily, and any
slot where something actually moved gets audited in full. It spends strictly
less and weakens nothing. I prefer it to the blanket cadence cut my predecessor
offered you, because that one halves oversight precisely when the system is
busiest and oversight is worth most. Overrule either way and I will record it.

**Unchanged heads-up from yesterday:** five of your eight open decisions have
pre-registered defaults firing on **2026-08-31**, the same morning your meter
resets. All five are reversible and none widens what is permitted, so this is a
crowded calendar rather than a risk — but answering one or two before Sunday
would thin it out.
