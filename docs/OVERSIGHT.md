# OVERSIGHT — 30th audit, 2026-08-25 12:40 UTC

## VERDICT: DRIFTING — the ledger is clean and the ratchets are green, but **the last iteration closed `rc=0` reporting a "healthy" pilot that was already dead**, and the scoreboard has not moved in five days while the denominator grew by eighteen

Three of the eight sections have **no findings**, and I say so first and plainly,
because a clean result is worth as much as a dirty one when it is true:

- **§1 ledger integrity — clean.** `run verify` re-judged all **83** auditable
  PASS entries from the record alone and probed **81** controls: 0 verdicts that
  no longer re-derive, 0 gates that ignore their control, 0 controls declared but
  never run, 0 gates that could not be replayed, 0 entries that could not be
  audited, 0 controls run but undeclared. Two PASSes have no control at all
  (`T0.01`, `T0.10`) — both existence claims, both long-declared, unchanged.
- **§2 thresholds and controls — clean. Not one threshold moved in the loosening
  direction in seven days.** Named and checked in §2 below; the only two edges
  that moved are *tightenings*.
- **§7 bakeoff hygiene — clean.** Three entries, one VOID correctly refused as a
  verdict, one winner with a recorded re-open trigger. One margin worth stating
  in the open (§7), declared rather than hidden, not a violation.

The three mandatory ratchets are **all green**: `coverage` exit 0 (0 uncovered
commitments, 0 CLAIM-DEAD, 4 known-dangling GOAL citations, shrink-only),
`decisions --check` "ratchet ok (5/10 undeclared)", `champions --check` "ratchet
ok (6/8 seats with a phantom arena)". That is the point of RANK 2 below.

---

## RANK 1 — the 12:07 iteration reported a running pilot that does not exist, and no organ noticed

**This is the finding with the shortest fuse and the longest history.**

The evidence, all of it mechanical:

| fact | value |
|---|---|
| iteration window | start `2026-08-25T12:07:09`, end `12:23:33 rc=0` |
| builder's closing report | *"The pilot is a tracked background task — I'll be re-invoked when it completes (no extra monitor needed; polling would be waste)… pilot running full-size on seed 90 (**pid 1552865, ~667 MB, healthy**)"* |
| `/data/sm03_pilot_seed90.json.log` | **0 bytes**, mtime 12:21 |
| `/data/sm03_pilot_seed90.json` | **does not exist** |
| `ps -p 1552865` (checked 12:38) | **no such process** |
| `experiments/tests/sm_03_nose_reports_occluded.py` | **untracked**, 32,086 B, mtime 12:20 |
| `docs/LESSONS.md` | **unstaged**, +35 lines (the `[s]`-tier entry) |

**Why it died.** The ladder loop runs `timeout 25m claude -p …`. A harness
"tracked background task" is tracked *for the lifetime of that session*. The
session ended at 12:23:33, so "I'll be re-invoked when it completes" is
structurally false for any job outliving the iteration — which is every job
worth detaching.

**The repair already existed and was not called.** `scripts/launch_detached.sh`
(written 2026-08-20, overseer 21st audit B1) exists for precisely this. Its own
header names the failure mode verbatim — *"reaped with the launching session ->
setsid"* — and closes four others, including *"'launched' believed on exit code
-> at 15 s the PROCESS must be alive and the LOG non-empty, or this script exits
1 loudly with the log tail."* It writes a `LAUNCH <iso> cwd=… cmd: …` header
line into the log **before** exec. The log is 0 bytes and carries no header
line: proof the guard was not the launcher. It was written after *"four detached
launches in 24 h died at import with exit 0 at the launch site and a 0-byte log,
and three iterations idled waiting on processes that did not exist."*

**This is the third recurrence of that class, arriving by the one route the fix
does not cover.** The prior instances died at import from a wrong cwd; this one
died with its session. `launch_detached.sh` closes both — but only if someone
chooses to call it. **A fix that lives in an optional tool is not a ratchet.**
And the new route is worse than the old one, because a harness "tracked
background task" *reads* as a stronger guarantee than a bare `&` while being
strictly weaker than `setsid`.

**Damage assessment, honestly bounded.** No false ledger row was written; no
threshold moved; SM.03's design is sound and traces cleanly to GOAL.md's smell
commitment (I read it — the supervised-readout redesign around SM.02's measured
learnability bottleneck, held-out layouts, hash-collision gate, and the
occluder-removed alive-proof are all right). The damage is to **throughput
honesty**: the system now believes it is waiting on a result that will never
arrive, which is the state that has cost it idle iterations before. Compounding
it, the entire 16-minute unit is orphaned in the working tree — the loop bans
`git add -A` and `harvest_bookkeeping` stages only `ledger.json` plus the two
GPU receipts, so nothing in the loop will pick up a 32 KB untracked test file.
The 13:07 iteration inherits a dirty tree that **no instrument names**.

**The generalisable lesson** (I have not appended it to `LESSONS.md` — see
"a note on what I did not commit" at the foot of this report):

> **A guard you have to remember to call is not a guard.** `launch_detached.sh`
> encodes four real failure modes and closes all of them — for the launches that
> go through it. The launches that matter are the ones an agent improvises under
> time pressure at the end of an iteration, which is exactly when it will reach
> for whatever is nearest. Worse, the nearest thing was a *harness feature whose
> name promises tracking*: "tracked background task" sounds like a stronger
> guarantee than `setsid` and is a weaker one, because the tracking is
> session-scoped and the session is the thing that is about to end. The durable
> repair is not another lesson; it is to make the *absence of a result* visible —
> nothing in this system watches for a launch that produced nothing.

---

## RANK 2 — 84 PASS for thirty-plus iterations while the ladder grew 169 → 187, and every ratchet went green doing it

The demonstrated count, straight from the log's own iteration headers:

```
83/169  (12 iterations)  →  84/179  →  84/181  (15)  →  84/183  →  84/187
```

- **The last PASS of any kind** was `NE.00`, 2026-08-24 06:28 — a `rule`-kind
  support spec, which `coverage` itself lists as *"support passing, not
  credited"* for hunger/thirst.
- **The last *claim*-kind PASS was `T3.01` (sight), 2026-08-20.** That is
  **five days with zero claim PASSes**.
- In those same five days the capability verdicts were: `T2.07` FAIL, `T3.07`
  FAIL, `LC.03 v2` VOID, `BA.02` VOID, `NE.01` FAIL (×2 attempts), `SH.01`
  ORACLE_CANNOT, `DP.05` FAIL, `T2.15` FAIL. **Eight consecutive red.**
- The demonstrated fraction fell from **49.1%** (83/169) to **44.9%** (84/187).

**The part the builder cannot see about itself, and which is my own role's
footprint.** Three of the last four working iterations spent their entire unit
on registration and bookkeeping: SH.02+SM.03 registration (09:07), LG.* family
registration (10:07), `goal_citations()` (07:07). Every one of those was an item
**my predecessors filed**, every one was owed, and every one was done well and
honestly. And every one is discharged by *writing registry text*. The scoreboard
they move is the ratchet, not the ledger:

| ratchet | before | after |
|---|---|---|
| `coverage` exit code | 2 (3 CLAIM-DEAD) | 0 |
| GOAL.md dangling citations | 5 | 4 |
| `champions` phantom arenas | 8 seats | 6 seats |
| registry size | 179 | 187 |
| **demonstrated** | **84** | **84** |

I am not alleging gaming, and I want that on the record: the builder refused to
manufacture a GPU dispatch four times this week when it had 29.7 free hours
expiring, registered only the LG specs the truncated research doc actually
contained rather than inventing designs to clear a count, and said so in writing
each time. That is the opposite of gaming. But **the instruments this role
installed now make "register a spec" the cheapest way to turn something green**,
and eighteen specs of denominator in two days against zero numerator is what
that looks like from outside. The correction is not to weaken a ratchet — it is
to add the one measurement none of them make: *claim-kind PASSes per week*.

---

## RANK 3 — three of the five "UNDECLARED" decisions were already answered by the owner, so the arming ratchet is measured against an inflated pool

`decisions --check` prints 10 open, 5 undeclared. Reading the file:

| entry the tool calls UNDECLARED | what its own body says |
|---|---|
| *"The owner's hands — how does a human TOUCH Jack's world?"* | **`DECISIONS_NEEDED.md:375`: "DECIDED 2026-08-09, same day: YES."** Tracked live at `INTEGRATION_QUEUE.md:445` as PENDING. |
| *"Was physics-first retired by argument instead of by bakeoff?"* | **`:534`: "DECIDED 2026-08-09: (a) RUN IT."** |
| `D3` | Superseded by the struck-through **"ANSWERED: YES (owner, 2026-08-10)"** section immediately above it at `:274`; the parser sees both headings. |

So the honest count is **2 genuinely undeclared open decisions, not 5**.

**And arming one exposed a guard-of-the-guard bug that makes three of the five
unarmable by construction.** `decisions.py:parse()` keys a heading with no
D-number by `title.split("(OPEN")[0].strip()[:52]` — a 52-character slice of
prose, spaces and all. But the declaration regex is
`_DECIDE = ^DECIDE:\s*([A-Za-z0-9._-]+)\s*$`, which **forbids spaces in an id**.
There is therefore no string you can write in a `DECIDE:` line that the parser
will join back to a title-keyed heading: *the entry cannot be armed at all*. Of
the five UNDECLARED entries, only `D3` carries a number; the other four are
title-keyed. So the standing duty "arm at least one per audit, the ratchet may
shrink and may never grow" has been pointed at a pool that was **80% unarmable**,
and every audit that reported those four correctly could not have discharged
them if it tried. I armed the credits question by giving its heading a number
(`D11`) — disclosed in a block quote at the heading itself rather than done
silently, since a heading renumber is one step beyond the strict append my
permissions name, and it is the only move that works. **Ratchet 5 → 4,
re-verified live.**

This matters in two further directions:

1. **The ratchet can be satisfied by bookkeeping.** "Arm at least one per audit;
   it may shrink and may never grow" is a good rule pointed at a denominator that
   is 60% stale. Moving three resolved entries out is correct and will shrink it
   — but it is not the arming the rule intends, and after the move the ratchet
   must re-baseline on the true 2 or it will read as slack it does not have.
2. **I checked the scarier reading and it is false, which is the good news.**
   "Was physics-first retired by argument" was decided *"(a) RUN IT — schedule
   the run after T2.01"* sixteen days ago, and `T5.01` is still `not implemented`.
   That looked like an owner ruling rotting unexecuted — the exact inversion of
   the D1 disease. It is not: `registry.py:759` gives `T5.01` `depends_on=["T2.01"]`
   and T2.01 is **FAIL**, so the ruling is honoured *by the dependency graph*, not
   ignored. The ruling is blocked in fact, and the file should say so.

---

## RANK 4 — four owner decisions default on the same day, six days out, and the largest one's default describes a bakeoff nobody has written

`D1` (**costs 38 specs** — blocks T2.01/T2.02, and through them touch, tool use,
proprioception, plasticity, sleep, social and generality), `D4` (8), `D10` (8),
plus `D7`/`D8`/`D9` — **all six `decide_by: 2026-08-31`**. That is six days away.

D1's armed default is not "do nothing": it strikes Option A as unconstitutional
and then says *"the remaining permitted arms go to a bakeoff at matched
experience, multi-seed, one pre-registered metric, learning gate and margin:
A-prime, B, C, D."* **That bakeoff does not exist as a spec, a test file, or a
queue row.** If 08-31 arrives unanswered, the default resolves the *constitutional*
half correctly and leaves the *work* exactly where it is — 38 specs still
blocked, now with the deadline spent. A default that fires into an unwritten
experiment buys a ruling and no motion.

---

## RANK 5 — W34 GPU: 0.31 h of 30 spent, five days to expiry, and the only live candidate is the pilot from RANK 1

| week | kaggle charged | of 30 |
|---|---|---|
| 2026-W32 | 21.06 h | 70% |
| 2026-W33 | 7.63 h | 25% — **22.4 h expired unused** |
| 2026-W34 (to date) | **0.3111 h** | **1%** |

The single W34 charge is `jack-ladder-1787631708` (T2.15, 0.3111 h, `ok: true`)
and it produced a real FAIL row with every gate green — **spend accounted for,
receipts committed, no waste**. `gpu_submissions.jsonl` joins cleanly: every
`attempt` has a `result`, no orphans.

`SM.03` is `GPU_SHORT` and was named the genuine dispatch candidate. It cannot
dispatch: its pilot is dead and its implementation is uncommitted, and
`dispatch.sh` refuses an unpushed HEAD. The waste here is real but it is
**downstream of the frontier being decision-blocked** (§RANK 4), not of
dishonesty — the builder has declined to manufacture a dispatch four times this
week and said so each time. I record that as correct conduct.

---

## §2 — thresholds and controls, seven days, examined by name

I diffed every commit since 2026-08-18 touching `experiments/registry.py`,
`experiments/registry_expansion.py` and `experiments/tests/`. Every numeric
constant that changed is either a **new spec's pre-registration** (T2.15, DP.05,
NE.00, NE.01, SH.02, SM.03, LG.*) or a **tightening**:

- `t0_21_coverage_audit_honest.py`: `N_PROPERTIES` **9 → 10** (`9449a1b`) → **11**
  (`7951f45`). The guard gained properties; each re-ran PASS and re-stamped.
- `registry_expansion.py` `ed2d969`: `DP.04.depends_on` **`["DP.00","VO.01"]` →
  `["DP.00","VO.01","LG.00"]`**. A **blocking edge added**, discharging an
  instruction that had stood in that spec's `notes` since 2026-08-10. This is
  the one diff hunk that superficially reads as a deletion; it is a line-wrap.
- `ne_01_nobody_survives_by_accident.py` `ddbe6b7`: `DELTA_T_NIGHT` **12 → 10**.
  A pre-run calibration, declared in the commit message with the sweep table
  (0.598 edge → 0.498 mid-band of an unchanged 0.3–0.6 gate), the table shipped
  in `metrics`, the gate untouched — and NE.01 FAILed anyway, twice, with
  identical digits. Sanctioned and correctly papered.

**No `_check` gained an `or`. No control was deleted or weakened. No seed count
was reduced. No assertion was removed.** DP.05 and T2.15 both carry explicit
"do not add seeds / do not re-dispatch unchanged" markers in their own
docstrings after FAILing.

## §3 — drift from the goal

**Nothing the builder did in the last day serves no GOAL.md sentence.** Item by
item: T2.15 harvest → *"Really learning, not appearing to learn"* + language
grounding; `goal_citations()` → *"If a piece of work does not trace back to this
page, question the work"*; SH.02 → *"too cold kills him… he builds a shelter"*
(GOAL.md:91–94); SM.03 → *"olfaction finds food, fire and decay… through
occlusion — the sense that works when sight fails"* (GOAL.md:45–48); LG.00/01/02/10
→ *"the proof he is a creature and not a costume"* (GOAL.md:167–170) and the
owner's LIAR TEST at :139–146. All four LG specs come from owner-designed
material that exists; the builder registered only what was written and left the
truncated §2.2–§11 as declared debt rather than inventing designs. That is the
right call and I want it recorded as such.

**The converse, which is the harder question.** From `coverage`: **14 of 23
commitments have live claim specs and nothing passing.** Of the three GOAL.md
claims most at risk of quiet neglect:

- **Curiosity** — 12 specs, **1 pass** (T2.08, coverage). `T5.06` "unprompted
  exploration is real" is `not implemented`. The ladder-and-apple sentence itself
  has no passing spec.
- **All-senses fusion / one brain in unison** — 21 specs, **1 pass**, and that one
  (`UB.9`) is registry-declared *conditional* pending a per-arm descent re-run.
  `T2.02` VOID, `UB.10` NOT_RUN, `T3.02`–`T3.06` not implemented.
- **Learning by living** — `death & retry` 3 specs 0 pass; `hunger/thirst` 5 specs
  0 pass; `shelter/building` 1 spec 0 pass; `fast/slow` 8 specs 0 pass. The four
  commitments closest to *"he gets thrown in, figures life out or doesn't, dies,
  and tries again"* have **zero** passing claims between them.

## §4 — is the builder alive and productive?

**Alive, disciplined, and not producing PASSes.** In the 24 h to 12:38: **24
hourly slots — 15 iterations started, 9 pace-skipped** on the `week:all models`
gate (31%→37% against a 8%→18% week-elapsed line, so the skips are the meter
working as designed, not a stall). **13 ended `rc=0`; 2 ended `rc=1`**, both on
2026-08-24 13:07 and 14:07 from session limits on every model in the fallback
chain — correctly detected, logged to `lost_iterations.log`, and *inherited* by
the 15:07 iteration ("inheriting 3 iteration(s) lost to limits"). That machinery
worked. **PASS delta over the window: 0.** No repeated identical failures, no
paused loop, no iterations aborting on load (max 0.16).

## §6 — stuck decisions

Covered in RANK 3 and RANK 4. Nothing is escalated to the owner that a
measurement could settle — **zero `MEANS-ESCALATED`**, which is the D1 disease
staying cured. Nothing overdue. **No owner decision was quietly acted on without
being recorded**: I checked the converse and D4's spend (the one the 29th audit
raised) is now recorded in D4's armed default as RATIFY-AND-CAP with the taking
dated 2026-08-13.

**I have armed one, per the standing duty** — the *"Claude credits are the
binding resource and are unmetered"* entry, open since 2026-08-09 and the one
that actually bit yesterday (three iterations lost to session limits). It is now
**`D11`**; the renumber was forced by the parser bug in RANK 3 and is disclosed
in a block quote at the heading itself. **Undeclared ratchet 5 → 4, re-verified
live.** Class `goal` (no experiment can settle how much may be
spent — the system cannot read its own balance). Default: **accept as-is**, which
picks the already-permitted action of changing nothing, and is justified because
the mechanical half of option (b) has since shipped — the pace gate (`e03693d`,
2026-08-24) reads `week:all models`, prints both meters, names which one governs,
and holds budget across the week; `lib_credits.sh` carries the fallback chain and
a 529 retry; lost iterations are logged and inherited. The status quo is no
longer "unmetered", it is "metered by cadence and acted on hourly". Reversible by
one owner line naming (b) or (c). `decide_by: 2026-08-31`.

## §7 — bakeoff hygiene

Three entries, all sound. `PS.01/J` recorded **VOID** for "arms below the 3.0σ
learning gate" and was correctly **not** treated as a verdict — it was re-run as
`J2`. `D2` was resolved by ledger replay with the method, the losing arm, and a
**re-open trigger** all recorded. One margin stated in the open rather than
buried: `PS.01/J2`'s winner `impact_speed` beats runner-up `peak_dvel` by
**2.66σ**, under the project's own 3.0σ ruler. The declared gate is over the
*null* (10.32σ) in `screen` mode, the arms are deterministic reductions of
identical cached rollouts with no training that could have failed — the file
says so explicitly in its screen rationale — and the runner-up margin is
*reported*, not gated. Declared, not hidden. Not a violation.

---

## §8 — THE HONEST SUMMARY

**No.** We are not closer to a curious humanoid that climbs the ladder than we
were yesterday, and we are not even closer to a longer list of green ticks — the
list of green ticks has been frozen at **84** for thirty-plus iterations. What
grew was the *denominator* (169 → 187) and the *quality of the instruments*.

That second thing is not nothing, and I will not pretend it is. This week the
system taught itself that GOAL.md's own citations must resolve, that a parked
spec is not coverage, that a champion's arena must exist, that a pace skip must
not block free bookkeeping, that a dead audit may not publish a green verdict.
Every one of those is a real ratchet and every one was earned by finding a real
hole. **The machine is the best it has ever been at telling the truth.**

But the truth it is telling is getting worse, and the five instruments now agree
on the diagnosis: `SH.01`'s ORACLE_CANNOT (a certified core, handed the answer in
its observation, sheltered in 0 of 27 lives), `LC.03 v2`'s single 3σ learner,
`DP.05`'s planners-eat-reactives-never, `NE.01`'s knife-edged shelter that either
freezes or cooks, and `T2.15`'s router losing to bag-of-words. **The world is too
shallow and the core cannot climb what gradient it has.** That is `D10`, it is
armed, and it defaults in six days.

The ladder-and-apple standard is the honest ruler, so apply it: nothing in the
ledger shows Jack trying to climb anything out of curiosity, falling, and trying
again. `T5.06` — *unprompted exploration is real* — is `not implemented`. Twelve
curiosity specs, one pass, and that pass is a coverage metric. The commitments
nearest *"he lives, he dies, he remembers"* hold zero passing claims between
them.

And the day's most telling fact is RANK 1: an iteration ended `rc=0` reporting a
healthy process that had already been reaped, with a 0-byte log, holding the only
GPU candidate of the week. Nothing in this system watches for **the absence of a
result**. Every instrument here reasons about rows that exist — the same blind
spot that let four commitments sit uncovered on 2026-08-10, wearing new clothes.

---

## FOR THE BUILDER

Ranked. **B1 is this iteration's unit; do not defer it.**

**B1 — Rescue the orphaned 12:07 unit, and make an absent result visible.**
Three parts, in order:
  (a) The working tree holds `experiments/tests/sm_03_nose_reports_occluded.py`
      (untracked, 32,086 B) and `docs/LESSONS.md` (+35 lines, the `[s]`-tier
      entry). **Semantically diff before you act** — this is your own timed-out
      unit, not damage — then commit both by explicit pathspec. Do not
      `git add -A`.
  (b) **Re-launch the SM.03 pilot through `scripts/launch_detached.sh`**, which
      is the tool that exists for this. Confirm the `LAUNCH …` header line
      appears in the log and the pid is alive at 15 s before you report anything.
      `/data/sm03_pilot_seed90.json.log` is 0 bytes and pid 1552865 is gone;
      treat the 12:07 launch as never having happened.
  (c) **The durable repair, and the point of the item:** add a guard that makes a
      launch-with-no-result *visible to an instrument*. The cheapest honest form:
      a `LAUNCHED:` receipt line (spec id, log path, pid, ISO timestamp) appended
      by `launch_detached.sh` to a single journal file, plus a check in
      `ladder_loop.sh`'s startup that reports any receipt older than N hours whose
      log is still 0 bytes or whose pid is dead with no ledger row. Today nothing
      in this system watches for the absence of a result, and this is the third
      recurrence of that class. Append the RANK-1 lesson above to `LESSONS.md` in
      the same commit as (a) — I deliberately did not write it there myself (see
      the note below).

**B2 — Fix `decisions.py`'s unarmable-entry bug, then move the three answered
decisions out.** Two parts:
  (a) **The parser bug first, because it is the one that silently disarms the
      duty.** `parse()` keys an un-numbered heading by a 52-char title slice
      (spaces included) while `_DECIDE`'s id class is `[A-Za-z0-9._-]+` (spaces
      excluded), so a title-keyed entry can never be joined to a `DECIDE:` block.
      Four of the five UNDECLARED entries were in that state. Either make the
      violation say so — `UNARMABLE: give this heading a D-number, it cannot
      carry a DECIDE block` — or slugify the title key and accept the slug in
      `_DECIDE`. Do not just renumber the remaining headings and call it closed;
      the *tool* must be unable to report an unarmable entry as merely unarmed.
      Add it as a known-answer case to whichever guard covers `decisions.py`.
  (b) Then move the answered entries out: *"The owner's hands"* (DECIDED YES
      2026-08-09, tracked at `INTEGRATION_QUEUE.md:445`), *"Was physics-first
      retired by argument"* (DECIDED (a) RUN IT 2026-08-09 — record that it is
      **blocked in fact** behind `T2.01` FAIL via `T5.01.depends_on`, so the
      ruling is honoured by the graph), and the superseded `D3 (original)`
      heading. Then **re-baseline `BASELINE_UNDECLARED` on the true count** — it
      is 10 today and the real pool after (a) and (b) is 2 — or the shrink will
      read as slack that was never won.

**B3 — Measure the thing RANK 2 is about.** Add **claim-kind PASSes per week** to
whatever `coverage` or `run status` prints. Today the ratchets all move when a
spec is *registered* and none of them move when a claim is *demonstrated*, so a
week of pure registration reads identically to a week of progress. One line of
arithmetic over `ledger.json` × `COVERS: … (claim)` closes it. This is the
counterweight to my own role's incentive footprint and I am asking for it against
myself.

**B4 — Write D1's bakeoff, or say in `DECISIONS_NEEDED.md` what firing its
default without one actually buys.** Six days. The default names four arms
(A-prime, B, C, D) at matched experience, multi-seed, one pre-registered metric,
learning gate and margin. None of that exists as a spec. 38 specs are downstream.

**B5 — Still open from the 29th audit, on its fifth carry:** UB.9's per-arm
descent measurement. Its PASS is registry-declared *conditional* and it is one of
only two passing claims under *one brain / unison*.

## FOR THE OWNER

Nothing here is urgent enough to interrupt you for, and none of it is a request
to change GOAL.md.

1. **Six decisions default on 2026-08-31** — `D1` (38 specs), `D4` (8), `D10` (8),
   `D7`, `D8`, `D9` — and I armed a seventh (Claude credits) onto the same date.
   That is one sitting, and it is the sitting that unblocks the frontier. `D10` is
   the one to read first: **five independent instruments now measure the same
   thing** — the world is too shallow to reward the behaviours we are asking Jack
   to learn, and the core does not climb the gradient that does exist. Its three
   branches are accept-one-learner, redesign W0, or redesign the arms.
2. **The credits question is armed with "accept as-is" as its default**, because
   the pace gate shipped and now meters cadence in a way that did not exist when
   you were first asked. One line naming option (b) or (c) reverses it.
3. **The honest state, in one sentence:** the measurement apparatus is in
   excellent health and the creature has not moved in five days — eight
   consecutive red capability verdicts, all of them informative, all of them
   pointing at the world rather than at the instruments.

---

### A note on what I did not commit

`docs/LESSONS.md` carries **uncommitted builder work** (the `[s]`-tier entry from
the 12:07 iteration). I found a generalisable lesson (RANK 1) and would normally
append it there, but committing that file would sweep in-flight work into an
overseer commit — the failure `c0afded` banned `git add -A` for, in both
directions. The lesson is written out in full under RANK 1 for the builder to
append alongside its own entry (B1). This commit stages **only**
`docs/OVERSIGHT.md` and `docs/DECISIONS_NEEDED.md`, by explicit pathspec. The
untracked SM.03 test file is left exactly where the builder left it.
