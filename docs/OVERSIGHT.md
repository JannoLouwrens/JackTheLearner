# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-18 06:37–07:2x UTC — 152nd audit.** (The Review's **Sunday FULL** ran
in this same slot — `review start — mode FULL, model opus, 40m / 240 turns` at
`06:37:04`, the 06:37 collision this desk keeps hitting. **It repaired this
audit's RANK 1 finding while I was writing it**, committing at `06:40:33`,
`06:42:33`, `06:43:21` and `06:44:12`. Memory lesson
`review-runs-during-the-audit` applied: every reading below is a re-run taken
after those four commits, the §1 I had drafted has been **withdrawn and
re-ranked** rather than shipped as live, and `git add` is by explicit name.)

## VERDICT: DRIFTING

**Not an integrity risk, and today is the evidence for that distinction: the
integrity machinery worked unprompted. A builder iteration left executable code
uncommitted for 15 hours while cron ran it, nine certificates were stamped
irreproducible — and the Review found it, inherited the work, committed it, and
routed the structural class on its own row, all before 06:45, without anyone
asking. The ledger's content is sound: 107 PASS, every commit resolvable, every
implementation importable, not one threshold moved in the loosening direction in
seven days.**

**What is drifting is where the capacity goes.** `107 → 107` across four
productive builder slots. Three of the owner's own constitutional commitments
have no live falsifier at all. Yesterday's and today's entire organ output —
builder and Review both — went into maintaining the measurement apparatus rather
than measuring Jack. That is not dishonest and it is not nothing, but it is not
progress toward a creature either.

Ranked by damage to the trustworthiness of the record:

1. **GPU dispatch is blocked right now by the desks' own normal operation** — a
   defect no organ has reported, live as I write. §1.
2. **`T0.02` keeps a `+dirty` certificate with no mechanical path to clear it**,
   falling through both regate lanes by 5.46 seconds. §2.
3. **An `rc=0` iteration that stops halfway is invisible to every monitor** —
   the cause of the whole incident, and the one part of it nobody has routed. §3.
4. **`PROGRESS.md`'s STALE banner states a reason that is now false.** §4.
5. **`coverage`'s `unreachable` floor has grown +2 beyond its last recorded
   reading with only one of the two increments named.** §6.

---

## §1 — RANK 1: the desks' own working files block the GPU path, for every spec

### Measured live, twice, during this audit

```
$ assert_ref_is_current()
RuntimeError: Uncommitted changes to tracked files -- the GPU would run
DIFFERENT code than you are testing:
 M docs/REVIEW_QUEUE.md
Commit and push first.
```

GPU dispatch was refused at two separate points in this sitting with the Review's
own queue page as the **sole** offending file — once at 06:5x and once at 07:0x,
between its commits. It is clear again now only because the Review finished and
committed. **The window is the defect, not the instant**: the predicate below is
unconditional, so the block recurs for the whole duration of every desk sitting.

```
is_code_dirt(' M docs/REVIEW_QUEUE.md')     = True
is_code_dirt(' M docs/DECISIONS_NEEDED.md') = True
is_code_dirt(' M docs/PROGRESS.md')         = True
is_code_dirt(' M docs/PROGRESS_LOG.md')     = True   (INSTRUMENT_INPUT_DOCS)
is_code_dirt(' M docs/CHAMPIONS.md')        = True   (INSTRUMENT_INPUT_DOCS)
is_code_dirt(' M docs/OVERSIGHT.md')        = False  (PROSE_DOCS)
is_code_dirt(' M docs/LESSONS.md')          = False  (PROSE_DOCS)
```

### The defect: a conditional class evaluated where there is nothing to condition on

`INSTRUMENT_INPUT_DOCS` is defined in `protocol.py` as — its words — *"Dirt for a
run of a spec that **DECLARES** it in `IMPL_DEPS`, and **clean for everybody
else**."* That per-spec conditionality is the entire reason the class exists
separately from `PROSE_DOCS`: it was carved after the 2026-09-02 cross-organ race
in which an audit's in-progress doc writes **VOIDed four PASS certificates**
(`PS.01`, `PS.02`, `PS.03`, `BA.01`) by accident.

`gpu.py:assert_ref_is_current()` is a **global precondition with no spec
argument**. It calls `offending_dirt()` → `is_code_dirt()` with nothing to
condition on, so the conditional class collapses to its strict reading and
applies to **every spec, including every spec that declares none of these
files**. `gpu.py`'s own docstring explains why the two predicates were unified —
*"there is now one predicate and zero permitted difference. T0.22 P15 pins them
together"* — and that unification is right for the question "did code move". It
is wrong here for a reason the unification did not anticipate: one caller knows
which spec it is asking about and the other cannot.

### Why this matters more than yesterday's incident

- **It fires every single day the Review sits.** A desk cannot dispose a queue
  row without an uncommitted `REVIEW_QUEUE.md` for the duration of its sitting,
  and `review.sh` FULL runs 40 minutes. For that window, no GPU job can be
  dispatched by anyone, for any spec.
- **The failure presents as infrastructure.** A dispatch attempted in that
  window raises `RuntimeError` naming a *markdown file* as the reason the GPU
  would run different code — which is the 2026-08-05 wrong-diagnosis failure
  (two GPU runs lost to stale code, diagnosed as an infra fault) arriving
  through the guard written to prevent it.
- **It is the 2026-09-02 scar at a boundary the fix did not cover.** That repair
  made the stamp conditional. It did not make the *gate* conditional, because
  the gate has no spec to be conditional on.
- **Nobody can see it from inside their own organ.** The Review cannot observe
  that its own working file closes the GPU path; the builder sees only a
  RuntimeError it did not cause. It is visible only from outside both, which is
  what this desk is for.

Honest bound on today's cost: **zero.** All three GPU classes read EMPTY/NOT
FILLABLE, the `T1.08` sixth-attempt prohibition forbids the only recent buyer,
and `run next` reads 0 fresh — so nothing wanted to dispatch. This is a latent
defect measured, not a loss incurred, and I rank it first on *reachability of a
wrong answer*, not on damage already done.

---

## §2 — RANK 2: `T0.02` has no mechanical path back to a clean certificate

Yesterday's incident (full account in the Review's two rows at `56c57e9`) left
**9** PASS rows stamped `+dirty` from `scripts/regate.{py,sh}`. State now:

| rows | state |
|---|---|
| `T0.05` `T0.08` | **already re-bought clean** by the 06:44 sweep |
| `T2.08` `T0.20` `T0.24` `W0.DIAG` `T0.28` `T0.31` | in the **stale lane**, which runs FIRST at `MAX_PER_RUN=8` — clear on the next tick |
| **`T0.02`** | **in neither lane** |

**The Review's judgement on the other six is correct and I verified it rather
than assuming it.** Its stamp says the nine *"are deliberately NOT re-bought by
hand — the rotation re-buys them at a clean commit on its own twelve-a-day
cadence, which is this row's whole thesis."* That reasoning is sound, and the
mechanism backs it: `cheap_stale()` returns exactly those six plus `T0.21`, and
the stale lane is drained before the rotation. They need no intervention.

`T0.02` is the exception, and the reason is arithmetic:

```
T0.02  budget label cpu<1min   (in CHEAP)       status PASS   commit 4a23405+dirty
       measured_child_seconds = 65.46 s   vs   label nominal = 60.0 s
       unsatisfied deps: []        in stale lane? False      in rotation? False
```

The rotation's inherited **cheap-by-MEASUREMENT** rule skips it — *"declares
`cpu<1min` but measured 65 s against the label's 60 s — not cheap until
re-labelled; stays with the loop"* — and the **stale lane cannot see it** because
its `impl_sha` never moved, which is the exact domain gap the rotation was built
to close. So `T0.02` ("Deterministic seeding") sits in the hole the new lane
exists to fill **and is excluded from the fix by 5.46 seconds**. "Stays with the
loop" means it waits for a builder slot, and the builder is pace-gated.

This refines the Review's call; it does not contradict it. Eight of nine are
genuinely self-healing.

---

## §3 — RANK 3: the `rc=0` slot that caused all of it is routed by nobody

The Review's new row, `regate-bought-nine-certificates-from-its-own-uncommitted-tree`
(OPEN, DUE 2026-10-24), is explicit that its question is *"NOT who left the file
dirty"* — it is about a lane that stamps `+dirty` on what it buys. Correct
scoping for that row. But it leaves the upstream cause unrouted, and I
reconstructed it from mtimes and commit timestamps because nothing on disk says
it:

```
2026-10-17T15:07:08  iteration start — 107/257 demonstrated, model fable
         15:12:38    commit ac51b29   t032 ruling option (i) executed
         15:16:03    commit 4a23405   t032 bill paid            ← LAST commit
         15:18:03    write scripts/regate.py                    ← AFTER it
         15:20:53    write scripts/regate.sh
2026-10-17T15:22:45  iteration end rc=0 — 107 -> 107 demonstrated
```

The slot was ordered by the 14:07 hand-forward to do *"(i) the two DUE-TODAY
rows, cheapest first … standing-pass is one added regate selection path"*. It
did the `t032` half, committed it, wrote the regate half, and hit its wall —
exiting **`rc=0`** with uncommitted code and **no `LOOP_JOURNAL.md` entry at
all** (the journal's newest entry is still the 14:07 slot).

**Every liveness instrument was green, and correctly so.** `rc=0` is the success
code. The pacer printed `0 failed slots` for 14 consecutive slots and was right —
nothing failed. `107 → 107` is legal for registration work. **This project's
monitors watch for failures; the expensive events are the successes that stopped
halfway.** The only on-disk trace was the missing journal entry, and
`LOOP_JOURNAL.md` is in `PROSE_DOCS`, so nothing gates on it — the closing step
that documents an iteration is the step a dying iteration cannot reach.

Two further consequences worth having on the record, both now moot but both
measured: the 8 commits between 17:02 and 06:44 carried a message advertising
*"the standing-PASS rotation"*, a lane `git show <sha>:scripts/regate.py` did not
contain — git history describing behaviour the repo lacked; and the lane burned a
slot when `T0.22`'s re-run came back `BLOCKED by T0.08 (PASS but stale — DIRTY)`,
so a dirty stamp on one spec refused an unrelated one. The Review found that
second one independently and recorded that no certificate was demoted by it.

Written up as a `LESSONS.md` entry this audit. **Not routed as a queue row by
me** — this desk may not register specs, and the Review's row covers the lane
half; the slot half is in FOR THE BUILDER below.

---

## §4 — RANK 4: the STALE banner's reason is false, its conclusion is right

`docs/PROGRESS.md`'s seal, stamped `2026-10-17T12:37:04` by `lib_seal.sh`:

> *newest row in docs/PROGRESS_LOG.md is 2026-10-13 (4d old; the schedule allows 1d)*

**The same run backfilled those rows 52 seconds later** — `ddcd8c6` (12:37:56)
wrote 10-14/15/16, `707ce9c` (12:42:10) wrote 10-17. `PROGRESS_LOG.md`'s newest
row is **2026-10-17**. The conclusion is still correct and load-bearing —
`PROGRESS.md`'s body is the 10-13 Review and is not current state — but its
stated reason is demonstrably wrong.

Ranked last because the damage is small. It is listed because it is the exact
shape of `audit-the-predecessor's-reason-for-not-acting`: a reader who checks the
reason finds it false and has every incentive to discount the conclusion with
it. A seal describes only its own file at the instant it fires.

---

## §5 — Thresholds and controls: NO FINDINGS, and I checked properly

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/` returns **two** commits.
Both move in the **tightening** direction:

- `32ed874` **NE.02 implemented** (first registration) — *adds* controls:
  `PLACEBO_MAX = 0.20` with a reverse gate that **VOIDs** rather than passes,
  `lambda_rescale_err_max <= 1e-9` as rig arithmetic, exercise gates that VOID
  not FAIL, two sabotage controls both verified CAUGHT, bars pre-registered
  before any number existed. Lands **BLOCKED** behind `NE.01` honestly rather
  than claiming a verdict.
- `8b74a34` **dp04 disposal** — *adds* GUARD (a2).

No threshold moved loosening-ward. No control deleted or weakened. No `_check`
gained an `or`. No seed count reduced. No assertion removed. **This is a true
"no findings" and I will not manufacture one.**

**Ledger integrity (audit item 1), re-derived rather than inherited:** 107 PASS
rows — **0** with an unresolvable commit (every `+dirty` sha resolves once the
suffix is stripped), **0** with a missing or unimportable implementation, 2 with
no declared `control` (`T0.01` repo-imports-clean, `T0.10` Kaggle round-trip —
both harness round-trip facts where a control is arguably N/A; noted, not
alleged).

---

## §6 — Coverage, decisions, champions, bakeoff hygiene

**`coverage` EXIT 2 — 0 commitments with NO declared spec, two ratchet floors
red.** The ladder is still the right ladder by the test that matters most, and
the exit code is the ratchet, not a broken tool. **Correction to my own first
reading this audit: I initially recorded `coverage` as EXIT 0, which was
`head`'s exit code through a pipe and not the tool's** — memory lesson
`never-tail-an-instrument`, paid again. The two red floors:

- **`unreachable` 98 of 257 (38%) vs baseline 95** — and against the last
  *recorded* reading of **96** (`ratchet_readings.json`, dated 2026-09-29, whose
  note explains 95→96 as `LT.02`'s honest demotion taking `LT.03` out of the
  reachable set). So it has grown **+2 beyond the recorded reading** and the
  reading was never refreshed. One increment is legitimate and named: `NE.02`'s
  first registration lands BLOCKED behind `NE.01` deliberately (the 14:07 journal
  says so, and `run blocked` confirms `NE.01=FAIL` blocks 8 and frees 7). I could
  not attribute the second. `coverage.py`'s own rule is that *"growth is
  permitted only with a named justification in the commit that grows it"* — so
  this owes either a justification or an unblock, and in the meantime the
  recorded reading understates the live number.
- **`pass_on_dead_dependency` 6 vs baseline 3** — six standing PASS rows resting
  on a dependency the ledger calls dead: `LF.02←T6.03 BLOCKED`, `T0.18←T0.13
  FAIL`, `T0.19←T0.13 FAIL`, `T1.12←T1.11 FAIL`, `T2.03←T1.08 FAIL`,
  `T2.14←T1.08 FAIL`. The floor was correctly **not** raised and each pair has a
  written cause. Noted as standing, owned debt, not a new finding.

The standing coverage hole is the honest answer to audit item 3's harder half:
**3 CLAIM-DEAD** commitments — every claim spec parked or foreclosed — and all
three are the owner's own words:

- **smell** (`SM.02` PARKED, `SM.03` FORECLOSED) — *"SMELL … not ornaments"*
- **shelter/building** (`SH.01` PARKED, `SH.02` FORECLOSED) — *"he builds a shelter"*
- **thermal (kills)** (same two) — *"too cold kills him, too hot kills him"*

Plus **14** commitments with live claim specs and nothing passing, including
`touch/contact`, `tool use`, `told world`, `proprioception`, `sleep`, and
`fast/slow` — where `DP.01`/`DP.02`/`DP.03` are all `welded<-LC.03`, a VOID.
This is `D42`, armed, `decide_by` **today**.

**`decisions --check` EXIT 0, ratchet 0/0 on all five classes.** No
`MEANS-ESCALATED` — no fork a measurement could settle sits on the owner's desk.
No `UNDECLARED`, so **there is nothing for me to arm this audit** and the ratchet
did not grow. 32 of 32 firings transcribed.

`D41` and `D42` reach `decide_by` **today**; both defaults are monotone HOLDs
that order no work. **Neither is OVERDUE yet** — on the `D43` precedent (due
10-16, fired 10-17 "on the first legal day") they fire **2026-10-19**. That is
the next audit's act, flagged so it is not missed.

**Six `CONDUCT-DESK` entries are STALE** — `D33` (25 d), `D35` (24 d), `D38`
(14 d), `D40` (8 d), `D39` (3 d), `D46` (1 d). These are "execute it, report it,
do not ask", and `decisions.py` lists them precisely *"so a stale conduct entry
cannot silently self-approve"*. Six, oldest 25 days, is the conduct docket
rotting. `D46` is the one correctly held: the Review answered the flag **in
writing** rather than self-approving, because exempting a desk organ from the
pace line re-allocates a shared account held with four paying tenants in favour
of the desk proposing it. That reasoning is right. The other five have no such
excuse.

**`champions --check` EXIT 0, ratchet clean on all six classes** — 0 phantom
arenas. Standing debt, all instrumented: 2 UNFALSIFIABLE (ASR, Speaker ID), 2
UNVERIFIED VERDICTS (**`Learning core` held BY VERDICT off `LC.03=VOID`** —
audit item 7's "VOID treated as a verdict", carried as the `D29` caveat;
**`World`** with no deciding run named and no `TRIGGER:` declared), 3 TRIGGER
DEBT, 4 unwinnable seats, 1 ARENA-UNREACHABLE (`Fast/slow coupling`, rooted at
`LC.03`).

**Bakeoff hygiene (audit item 7): no findings.** `SO.10` is recorded as a
**TIE**, not a winner — *"laplace-full leads laplace-w30 by only 0.26 sigma
(margin 1.5). The choice does not matter yet; taking the cheapest tied arm."* A
result inside the noise margin, labelled as such and not promoted: the rule
working. `LG.13`'s `meaning-mass` win carries its screen rationale declared
before any arm number existed. No decision made without a learning gate.

---

## §7 — The routed work, the builder, liveness, compute

**`run review-queue` EXIT 0 — OVERDUE 6 → 0 during this audit.** Readings taken
in sequence on the same tree as the Review worked: **6** at 06:37 (audit start),
4 after `56c57e9`/`134f92a`, 2 after `96d046c` (t205 RULED), **0** after
`c77e45f` ("the bound pair ruled in ONE sitting"). 142 routed, oldest live 49 d.

**This is the most productive desk sitting in a fortnight and the number earns
saying so.** Two of the six were not undone work at all —
`standing-pass-certificates-are-falsifiable-only-by-running-them` was executed
but uncommitted (ACTED at `4a3e646`, §1–§3) and `t032` was executed at `ac51b29`
awaiting a stamp — but the other four were genuine breaks, including `t205` on
its third and two on their second, and all four were dispositioned by ruling
rather than re-dated. `pl02-void-gate-quantifies-over-its-own-nulls` was on its
first re-date.

**The number above is stale by construction and the next audit must not inherit
it** — the live reading is `run review-queue`. I record the trajectory rather
than a snapshot precisely because a snapshot taken at 06:37 would have reported
6 violations on a morning the desk cleared all of them.

Throughput says the desk is the constraint: **arrived 3, disposed 8, designed 7**
over 7 cycles — **104 cycles to clear 74 live rows** at the measured net rate.

**Schedule half.** The Review completed no sitting from 10-14 to 10-17:
10-14/15/16 had `06:37` plus all four `--retry` polls refused `STOPPED at 100%
weekly usage`; 10-17 started 12:22 and died **`rc=124`** at 12:42. The holes were
backfilled by that run, which also diagnosed its own cause — `review.sh:180`'s
labelled-hole fallback is gated on `[ "$RC" -ne 0 ]` and the deferral path
`exit 0`s at `:60` before reaching it, routed as
`review-defers-leave-no-labelled-hole`. Correctly diagnosed, not yet repaired.
**Today's FULL sat and is the most productive desk sitting in a week.**

**Builder.** Not the 119-dark-slot blackout the stale `PROGRESS.md` describes —
the shared meter moved and the builder ran **four productive slots on 10-17**
(11:07, 13:07, 14:07, 15:07), all `rc=0`. Then `16:07 ABORT: load 10.47 above
6.0` (correct — tenant protection), then **14 consecutive PACING skips**,
`0 failed`, 14.7 h since the last `rc=0`, `66%` against a `61%` line. PASS delta
across the four productive slots: **107 → 107**. The work was real (NE.02
registered, GPU ref-pin at `c08d15a`, `t032`) and bought no new certificate,
which is honest and expected for registration-and-harness work.

**Compute honesty (audit item 5).** CPU 2,982.8 s (10-17) and ~2,993.7 s (10-18
so far); most of today's is the regate lane re-buying what §2 describes. GPU:
`gpu_budget.json` has **no `2026-W41` and no `2026-W42` entry at all** — W41's
30 h expired with 0.000 drawn and W42's 30 h expire today. **GPU-hours spent with
no ledger entry to show for them: none this week, because none were spent.**
Said against urgency as `D40` requires, and verified independently rather than
inherited: those hours had no designed buyer — all three GPU classes read
EMPTY/NOT FILLABLE and the `T1.08` prohibition forbids the only recent one. The
expiry is not a loss caused by §1.

---

## §8 — Drift (audit item 3) and the honest summary (audit item 8)

**No drift.** Everything the builder touched traces to GOAL.md:

| work | GOAL.md sentence |
|---|---|
| `NE.02` need×ablation matrix | *"the needs ARE the curriculum"*; *"components that must EARN their parameters via ablation or be deleted"* |
| GPU ref-pin (`c08d15a`) | *"protects the honesty of watching what happens when the three meet"* |
| `t032`, the regate rotation | *"Really learning, not appearing to learn"* — a certificate nothing re-runs is not evidence |

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
No.** `107 → 107`. Smell, shelter and cold-kills — three commitments the owner
wrote himself — have no live falsifier, and have not for weeks. Curiosity has
12 specs and 2 passing; all-senses fusion has 28 and 1.

**But today is genuinely the best day this system has had in a fortnight, and it
is worth being precise about why, because it is not a green tick.** Yesterday
afternoon the builder was let back on the box, worked four slots, and in the last
one wrote the repair for a row whose entire subject is *"a standing PASS is
falsifiable only by running it"* — then exited `rc=0` without committing it. For
fifteen hours that uncommitted code ran on cron, buying certificates out of a
tree dirtied by its own source, stamping nine of them irreproducible, closing the
GPU path, and reporting itself to the queue as **not done**. Every monitor was
green and every monitor was right.

Then the Sunday Review sat, read `regate.log`, found it in its own `git status`,
inherited the work, committed it, verified the lane clause-by-clause against the
code rather than its docstring, stamped the row, and routed the structural
residue on a new row with a four-option menu and nothing ordered — in seven
minutes, unprompted, while telling itself explicitly that *"the stamp must not
swallow what committing it repaired."*

**That is what this project is actually building, and it worked.** A system that
can be wrong for fifteen hours and then catch itself without being asked is
worth more than a system that is never wrong because nothing runs. The ledger is
not closer to a creature today. The apparatus that will one day have to tell us
honestly whether he climbed the ladder is measurably more trustworthy than it
was yesterday — and the three things still wrong with it are in §1, §2 and §3,
none of which any organ could see from inside itself.

---

## FOR THE BUILDER — ordered, cheapest and most load-bearing first

1. **`T0.02` is the one `+dirty` certificate with no mechanical path back
   (§2) — re-buy it by hand, in-slot, and price the bill first.** It measures
   65.46 s against its `cpu<1min` label's 60 s, so the rotation skips it as
   not-cheap, and the stale lane cannot see it because its `impl_sha` never
   moved. Do **not** "fix" this by widening `CHEAP_NOMINAL_S` or re-labelling the
   spec to make the skip go away — the measured-cheapness rule is correct and
   was bought by the 128th audit's day-meter foreclosure. The honest options are
   (a) one manual re-buy now, or (b) relabel `T0.02` to its true class
   `cpu<10min` **with the measurement quoted in the commit message**, which puts
   it in the rotation permanently. (a) is ~65 s and closes it today; (b) is the
   durable one. Taking (b) is a threshold move and must say so.
2. **Do not touch the other six — they self-heal, and I verified it.**
   `cheap_stale()` returns `T2.08`, `T0.20`, `T0.21`, `T0.24`, `W0.DIAG`,
   `T0.28`, `T0.31`, the stale lane drains before the rotation at
   `MAX_PER_RUN=8`, and the tree is clean as of `4a3e646`. `T0.05`/`T0.08`
   already cleared at `deb311f`. The Review's "deliberately NOT re-bought by
   hand" call is right; re-buying them manually would spend CPU to prove the
   lane you just committed does not work.
3. **§1 needs a design, and it is NOT mine or yours to just patch** — route it
   or ask the Review to. The defect: `gpu.assert_ref_is_current()` has no spec
   argument, so `INSTRUMENT_INPUT_DOCS`' deliberate per-spec conditionality
   (*"clean for everybody else"*) collapses to its strict reading and blocks
   dispatch for every spec whenever any desk holds an uncommitted queue/decisions
   /progress/champions page. Verified live this audit. **The shape of the repair
   is a conditional gate, not a wider exemption** — do not add these files to
   `PROSE_DOCS`, which is fork (a) and was already refused twice on the record
   (it trades the accidental-block trap for unstamped drift in instrument
   *inputs*). `assert_ref_is_current` taking an optional spec and passing it to a
   spec-aware predicate is the additive version; `T0.22 P15` pins the two
   predicates together, so that pin is part of the design, not an obstacle to
   route around.
4. **Write the 15:07 slot's missing `LOOP_JOURNAL.md` entry**, retrospectively
   and labelled as such (§3). The gap is currently the only on-disk evidence
   that the slot ended mid-unit, and reconstructing it cost this audit an hour of
   mtime archaeology.
5. **Account for `unreachable` 96 → 98, or unblock it (§6).** The recorded
   ratchet reading is 96 and dated 2026-09-29; live reads 98 against baseline 95.
   `NE.02`'s deliberately-BLOCKED first registration is a legitimate named reason
   for one increment and is already written in the 14:07 journal — **so the cheap
   half of this is just `run ratchets record` with the cause quoted**, which the
   10-17 slot did once and has not done since. The second increment I could not
   attribute, and that is the half that needs a name. **Do not raise
   `UNREACHABLE_BASELINE`** — `coverage.py:1186` already carries the standing
   sentence that a builder may not raise the floor over its own organ's edit.
5. **`HR.1` family-fate is NOT builder work.** The 13:07 slot established arm
   (a) was implemented and REFUTED and arm (c) executed FAIL (`1f5575e`, all
   2026-09-23), with `BUILDER-TRACE` on the row. Do not re-execute it on the
   strength of the stale `PROGRESS.md` FTB 2, which was written blind to that —
   memory lesson `progress-items-may-be-discharged-elsewhere`.
6. **`T1.08`: the sixth-attempt prohibition is live.** Do not dispatch.

## FOR THE OWNER

**1. NO DECISION NEEDED — but the thing I would want you to read is a success,
not a failure.** Yesterday a builder slot stopped halfway and exited with the
success code, leaving live code uncommitted; for fifteen hours a cron lane ran it
and quietly degraded nine certificates. No monitor fired, because none of them
watches for a success that stopped halfway. This morning the Review found it on
its own, repaired it, and wrote down what it must not pretend it had fixed. The
apparatus caught itself. That is the property this project is actually trying to
buy, and it is now demonstrated rather than claimed.

**2. NO DECISION NEEDED — GPU dispatch is currently blocked by a markdown
file, and the cost today is zero.** `experiments/gpu.py` refuses any dispatch
while any desk has an uncommitted `REVIEW_QUEUE.md`, `DECISIONS_NEEDED.md`,
`PROGRESS.md`, `PROGRESS_LOG.md` or `CHAMPIONS.md` — so every day the Review
sits, the GPU path is shut for every spec. **No hours were lost to it**: all
three GPU classes are EMPTY/NOT FILLABLE and the `T1.08` prohibition forbids the
only recent buyer, so nothing wanted to dispatch. `2026-W42`'s 30 free Kaggle
hours expire today with 0.000 drawn, as W41's did, and that is for the reasons
`D40` already covers — not for this one. Routed to the builder as a design
question in §1/FTB 3.

**3. NO DECISION NEEDED — `D41` and `D42` reach `decide_by` today and fire
tomorrow.** Both pre-registered defaults are monotone HOLDs: they move no
threshold, repair nothing cosmetically, spend nothing, and leave every finding
standing and red. `D41` leaves the two-artefact recipe divergence live and the
`demonstrated` count reported without its caveat. `D42` leaves **smell**,
**shelter/building** and **thermal (kills)** — three of your own constitutional
commitments — with no live falsifier, and says so plainly. Both entries state
that they are deliberately *not* the recommendation, because choosing which of
your commitments matters most is not a desk's call. One sentence from you
overrides either, today or later.

**4. NO DECISION NEEDED — the conduct docket is rotting, reported rather than
cleared.** Six `CONDUCT-DESK` entries are stale, oldest **25 days** (`D33`).
These are desk-executable by their own class — "execute it, report it, do not
ask" — and they are not being executed. `D46` (the reporting floor: desks exempt
from the pace line, still bound by the 90 % hard stop) is correctly held, because
exempting the desk that proposed it re-allocates a shared account you hold with
four paying tenants. The other five have no such excuse. This desk cannot execute
another organ's conduct items; the Review can, and until this morning had been
dark since 10-13.
