# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **102nd audit — 2026-09-19, 06:37–07:0x UTC.** Opened at HEAD `4509457`,
> **concurrently with the Review's own 06:37 DAILY sitting** (the known
> collision), which committed five times while I read — so every number below
> was re-derived after `8b10fd7` and the two readings that moved are named as
> before/after pairs. `demonstrated` **108/249 (43.4%)**, flat since
> 2026-09-14T04:18 — **5 d 2 h**, the sixth consecutive zero-net day.
> Builder, last 24 h: **19 iterations, 17 × rc=0, 2 × rc=1** (both session-limit
> 15:07/16:07 yesterday, recovered at 17:07), **12 commits** since the 101st
> audit's commit `8a21968`.

## VERDICT: DRIFTING

Two things are true and neither cancels the other.

**The ladder moved, for the first time in days, and it moved on the apple.** The
04:0x–06:0x slots re-scoped and implemented `LT.01`'s C2' — the Ladder Test's own
honesty certificate, `GOAL.md:31-33`, the ladder-and-apple sentence itself — with
an adversarial adhesion-enabled height-seeker that reached **0.7142 m against the
0.6 m bar with 0 ladder engagements** while the random null read **0.0099 m**.
That is science, it is the right science, and the three preceding audits were
right to keep saying the board had none.

**And the run that would have banked it is dead on the floor, nobody knows, and
four of the owner's own constitutional commitments still have zero specs while
the builder spent four consecutive slots calling the board empty.**

Ranked by damage to the trustworthiness of the ladder:

| # | finding | damage |
|---|---|---|
| 1 | **4 constitutional commitments have NO spec** (`heavy`, `far`, `tiring`, `worth-it`) while four slots in a row journalled "the board is genuinely empty" — verified on four instruments, none of which can see a missing spec | the 2026-08-10 miss, recurring *inside* the file written to prevent it |
| 2 | **`LT.01` attempt 2 is dead and no organ knows** — launched 06:09 as a session child, session ended 06:09:57, no process, no artifact, no commit, no journal. **Fifth** dies-with-parent occurrence, **52 minutes after the lesson naming it the fourth was committed** | a lost run that the log asserts is in flight |
| 3 | `review_queue_violations` **10 → 12 recorded at 06:43**: +4 clock (four more desk rows broke at midnight), −2 act. The act justifies 2 of the 6 points of movement; the MOVED banner is now quiet and the four broken promises are **baseline** | a red number entering its own floor |
| 4 | Queue **12 OVERDUE** (was 14 at 06:38), oldest live 26 d, drain still **UNBOUNDED** — but the desk is mid-remedy and it is the first reduction since 09-13 | `D28`'s subject; improving |
| 5 | **27.7837 free Kaggle GPU-hours expire tonight** with no legal buyer — third consecutive weekly write-off | real, and refusing to manufacture a buyer is still correct |

No silent loosening. No ledger integrity defect. Sections 1, 2, 6 and 7 are
clean and I re-derived them rather than inheriting the 101st audit's word.

---

## RANK 1 — Four of the owner's own commitments have zero falsifiable claims, and the builder has journalled "the board is genuinely empty" on four consecutive slots. Every instrument it named is structurally blind to a missing spec. That is the exact miss `coverage.py` was built for.

`$PY -m experiments.coverage`, first four rows, **EXIT 2**:

```
heavy      0 specs  0 pass  0 now  1 nominated   NO SPECS   ^ GOAL 187: survival-earned primitive
far        0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187: survival-earned primitive
tiring     0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187: survival-earned primitive
worth-it   0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187: survival-earned primitive
```

These are `GOAL.md:186-188`, the sentence that justifies the entire survival
programme: *"Survival earns him the primitives that make anything else mean
something — hot, heavy, far, tiring, dangerous, worth-it, that-person-lied."*
Three of the seven are in the register. Four have **no claim spec at all** — not
blocked, not parked, not foreclosed. Absent.

**What the builder did with that fact, in its own words.** The 02:0x journal
(`1920cff`) and the 02:0x ladder log:

> "**The board is genuinely empty**, re-verified rather than inherited: `run
> next` offers 0 fresh of 44 (only settled verdicts); `run coverage` reads 5
> empty cost classes with no path in, every blocked repair a Review redesign;
> `run blocked`'s terminal blockers are all Review-owned; `run stale` lists 12
> stale claims… zero stale PASS certificates."

Four instruments, honestly run, correctly reported — and **not one of them can
see a spec that does not exist.** `run next` ranks ids. `run blocked` walks the
dependency graph between ids. `run stale` compares hashes of implementations.
`coverage`'s *cost-class* section, which the journal quotes, reports on classes
with no runnable *spec*; the commitment table above it, which reports on
commitments with no spec at all, was not quoted. This is the 2026-08-10 miss in
the audit script's own words — *"a missing spec has no id, appears in no `run
blocked` ranking, blocks nothing and fails no gate"* — recurring four days
running, inside the very register built to end it.

**And nothing will go red about it, by construction.** `8f0f772` (101st audit
RANK 2, correctly executed) made this a counter — and `run status` now reads:

```
commitments_uncovered = 4  (unchanged since 2026-09-19)
  vs declared floor 4: AT floor — ok
```

**"AT floor — ok" for a four-item hole in the constitution.** The convention is
right (`unreachable` 97, `gpu_unattributed_jobs` 21 and `fail_unowned` 0 all
baseline at their creation value), the execution is right, and the consequence is
that the hole is now a *number* rather than a *violation* — permanently green
until someone shrinks it by hand. That is not an argument against the counter. It
is the reason this audit, and not an instrument, has to be the thing that shouts.

**Who may close it, precisely — because the queue row says the wrong thing.**
`goal-187-names-seven-primitives-four-have-no-commitment` (OPEN, DUE 2026-09-18,
**+1 d overdue**) splits the work in two and routes the second half away:

> (ii) REVIEW, then possibly the OWNER: once visible, does this project COMMIT to
> a falsifiable claim for heavy / far / tiring / worth-it, or is `GOAL.md`'s
> sentence corrected to name only what it intends to test?

Half (i) — add the four to the register — **shipped 09-18** and is why the number
reads 4. But the fork as framed is not symmetric: **striking the words is the
owner's; registering the claims is already permitted and already routine.** The
builder registered **five** specs on its own authority in the last seven days —
`HR.1` (`5283aad`), `LG.12` (`939e3c4`), `LG.13` (`848d452`), `SO.10`
(`5c2c93e`), `W1.02` — each "Register and implement", none of which needed a
ruling. A fork with one arm that any organ may walk today does not belong on
another desk, and `PROGRESS.md` item 6 escalating it to the owner ("**the fork
… remains yours**") is the `D1` shape: a question a *registration* settles,
waiting on an argument.

**A spec registered blocked still closes this hole.** `heavy` needs mass-varying
objects, `tiring` needs fatigue coupling, `far` needs distance cost, `worth-it`
needs a value trade-off — several of those world features do not exist yet. A
claim registered today would land BLOCKED or gate-provisional, which is
*strictly better than absent*: a blocked claim has an id, appears in `run
blocked`, joins `unreachable`, and names the world feature it is waiting for.
That is what `claim_dead` and `unreachable` are FOR.

**One trap, stated so the cheap wrong repair does not get taken.** `coverage`
prints `nominations (declare or ignore): PG.2` against `heavy`. `PG.2` is
**"Water works: buoyancy + drag"** — *"a passive ragdoll floats at the
equilibrium depth its density ratio predicts (±10%)"*. It is a keyword artifact
(density/mass), not a claim about weight as a learned survival cost. Declaring
`COVERS: heavy` on it would take the counter 4 → 3 by **false credit** — the
exact defect that had `T6.03` cited for a `GOAL.md` sentence it did not test for
36 days. `far`, `tiring` and `worth-it` have **zero** nominations, honestly.

**My own constraint, declared.** This audit's FIRST rule is *"If a commitment is
uncovered: register a spec for it that iteration, before anything else."* I MAY
NOT register a spec. So the rule's executor is the builder, and it is item 1
below.

---

## RANK 2 — `LT.01` attempt 2 was launched at 06:09, the session ended at 06:09:57, and the run died with it. Fifth occurrence of a class whose fourth occurrence was written into `LESSONS.md` fifty-two minutes earlier, by the previous slot, about this same spec.

**What the log says is happening** (`ladder.log`, 06:0x slot, the last line in
the file):

> "The run's completion will re-invoke me directly (the background task from
> 06:09), so no extra monitor is needed. Waiting for LT.01 attempt 2 to finish
> (~06:43 projected); on landing I'll read the branch that fired, commit the row
> by name, stamp `lt01-c2-body-cannot-rise` ACTED, and journal."
>
> `2026-09-19T06:09:57+00:00 iteration end rc=0 — 108 -> 108 demonstrated`

**What is actually happening — four independent instruments, none of which the
builder can run about itself after it has exited:**

| check | reading |
|---|---|
| `ps -eo pid,ppid,lstart,etimes,args` filtered to `/data/venvs/jackthelearner` | **no match** — no process |
| oldest non-kernel process started after 06:00 | `overseer.sh`/`review.sh` at **06:37**; nothing survives from 06:09 |
| newest artifact under `/data/` | `t108_backend_probe.json`, **2026-09-18 13:09** |
| `git log` / `git status` for the 06:0x slot | **zero commits, clean tree, no journal entry** |

The run was projected at **~34 min** for 3 seeds. It got **48 seconds**.

**Why this is the fifth and not the fourth.** `LESSONS.md` (committed `4509457`,
**05:18 today**) records occurrence four — the 04:0x slot launching this same
pilot as a session child with 90 s of runway — and states the rule:

> "**you may not launch what you cannot outlive.** Before starting any local
> computation, compare its projected runtime … against `JACK_ITER_DEADLINE -
> now`, and if it does not fit with margin for the commit, the correct act is …
> commit the implementation clean, write the projection into the handoff, and
> let the next slot spend its full budget on the run as its first act."

**The 06:0x slot obeyed the arithmetic and lost anyway,** and the distinction
matters for the repair. 06:09 + 34 min = **06:43**, inside the slot. The
deadline test *passed*. What failed is the **lane**: the lesson is about *time*,
and this instance is about *foreground vs background*. The slot handed the run to
a background task and then ended its turn believing "the run's completion will
re-invoke me" — a mechanism that only exists while the session is alive. Once
`claude -p` returns, `ladder.sh` closes the iteration and cron opens a fresh one
an hour later. **Nothing re-invokes anybody.** The lesson as written does not
forbid what happened; it must be extended, not merely re-cited.

**Damage, bounded but real.** `b16de57` (the implementation) is committed, clean
and pushed, and the pilot record lives in its docstring — nothing is lost but the
attempt. `lt01-c2-body-cannot-rise` stays live (DISPOSITIONED, DUE 2026-09-24),
so the clock has five days on it. The compounding cost is the one the lesson
already names: **a slot that leaves no commit leaves no journal**, so the 07:0x
slot inherits a clean tree, a correct 05:0x handoff, and a log line asserting a
run is in flight that is not.

**And the structural point, which is why this is RANK 2 and not a footnote:
no organ in this system watches for the absence of a result.** The ladder log
records intentions; the ledger records outcomes; nothing compares the two. An
iteration that ends saying "waiting on X" and is never re-entered produces
exactly this: a promise in a log file, no process, no artifact, no row, and
silence from every gate. It took a `ps` and an `ls -lat` to find, and only
because this audit happened to open 28 minutes later.

---

## RANK 3 — `review_queue_violations` went 10 → 12 into the recorded baseline at 06:43. Two of the six points of movement were bought by an act; four were the clock. The MOVED banner is now quiet, and four broken promises are floor.

`d2a6580` (Review, 06:43), and the commit message is fully honest about what it
contains — I am not alleging concealment:

> "review_queue_violations 10 -> 12. Two movers, opposite signs … **+4 was the
> midnight rollover of four more DUE-2026-09-18 desk rows going OVERDUE (a CLOCK
> movement, nobody's act, and the fourth consecutive night this desk has paid
> it)**, and -2 is this sitting disposing waits-on-declared-field and
> hash-salt-lottery-in-a-gated-metric."

Before (06:38, my first run): `review_queue_violations = 14 !! MOVED +4 since
2026-09-18 (was 10)`, 14 OVERDUE printed by name.
After (06:45): `review_queue_violations = 12 (unchanged since 2026-09-19)`,
**no banner**, 12 OVERDUE printed by name.

**The −2 is earned and I checked it row by row.** `5206753` and `75acde7` each
deliver a written design, hand the implementation half to the builder, and
re-date 09-17 → **09-21** with the reason stated on the row and the date taken
from `review-queue`'s own `next_free_due` print. That is one of the three honest
repairs, it is the desk's first reduction of this counter since the thirteen
broke together on 09-13, and it landed **two days before `D28`'s default makes
it compulsory.** Credit where it is due.

**The +4 bought nothing and is now invisible.** Recording absorbs it. A reader
tomorrow sees `12 (unchanged)` and has to open a commit message from an hour of
one morning to learn that a third of the number is four promises that broke while
nobody was at the desk. The instrument **already knows how to make this
distinction** — `ef2757c` split this family's delta into CLOCK and ACT, and
`review_queue_net_arrivals` prints `(clock +0, act -1)` on every status run.
`ratchets record` does not use the split: it writes one scalar.

The 101st audit ordered this counter left unrecorded, and the 03:0x builder slot
obeyed. The Review is not bound by a `FOR THE BUILDER` item and its record is
narrated in full, so this is **not** a violation by anyone. It is a gap in the
recorder: **a clock-driven rise can currently be absorbed into a floor by a
legal act.** The repair is item 3 below and it is small.

---

## RANK 4 — the queue: 12 OVERDUE, drain still UNBOUNDED, and for once the direction is right

`$PY -m experiments.run review-queue`, **EXIT 2**, after the Review's sitting:

```
30 OPEN, 3 HELD, 13 DISPOSITIONED, 16 ACTED, 0 DECLINED of 62 routed
oldest live 26 d; consumer last ran 2026-09-19 (0 d ago)
arrived 16 (2.29/cycle) | disposed 8 (1.14/cycle) | designed 2 (0.29/cycle)
drain UNBOUNDED — 46 live rows, arrivals exceed disposals by 8
12 VIOLATION(S) — OVERDUE 12
```

The twelve, with the age of the broken promise: `w1-world-edit-window` (+1 d),
`t211-diayn-metric-cannot-separate-mi-from-noise` (+3), `aggregate-hides-worst-seed`
(+1), `five-commitments-are-claim-dead-behind-foreclosures` (+3),
`hr5-fixture-refuted` (+3), `goal-187-names-seven-primitives-four-have-no-commitment`
(+1 — **RANK 1's row**), `so07-recording-worlds-fail-the-reference-bar` (+1),
`t108-noise-floor-is-quoted-by-nobody` (+3),
`so10-tie-break-hands-the-seat-to-an-ineligible-arm` (+2),
`lg13-champion-makes-lg10s-invariance-conjuncts-structural` (+2),
`oversight-for-the-builder-has-no-reader` (+2),
`a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere` (+1).

Two things worth separating. **`DISPOSITIONED` is not a disposal** — the
instrument counts only `ACTED`/`DECLINED` as leaving the live set, and both of
this morning's two rows are still live and still ageing on a 09-21 clock. So the
*drain* has not moved; what moved is the *violation* count, by legal re-dating.
And `coverage`'s `FAIL-OWNED-BUT-UNDRAINED` still reads **24 settled FAILs whose
only repair owner is a row on this desk**, against `fail_unowned = 0` at floor.
Both halves remain true at once, exactly as `D23` framed it.

**Sunday's FULL is tomorrow, and the desk has staked a public attempt on it**
(`PROGRESS.md` item 7: *"dispose the overdue class first and in bulk, ruling
rather than re-dating, before Part 2 opens a single certificate"*). One fact for
whoever reads the result: `D27`'s own text records that the Review's FULL *"has
died at max turns on four of four Sunday FULLs"*. The remedy is scheduled into
the least reliable sitting this system owns. That is not a finding against the
desk — it is the thing to check tomorrow before believing either outcome.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, re-derived not inherited.**
149 rows: **PASS 108, FAIL 26, VOID 14, BLOCKED 1**. All 108 PASS ids resolve in
`BY_ID`. All 108 `commit` fields resolve under `git cat-file -e <sha>^{commit}`
— **zero unresolvable**. All 108 declare a `control`; **106 carry
`control_metrics`**, and the two that do not (`T0.01`, `T0.10`) declare
`control = "NONE, BY DECISION (52nd audit B5)"` with the argument written on the
spec. Nothing on the scoreboard is a lie.

**2. Thresholds and controls over time — ONE structural change, JUSTIFIED, and
net-strengthening.** `git log -p --since="2026-09-12"` over `registry.py`,
`registry_expansion.py`, `tests/`. The only clause that changed direction is
`LT.01`'s **C2 → C2'** (`b16de57`, today 05:17), and it deserves naming loudly
because on its face it is the shape this section exists to catch: a clause that
could FAIL became a clause that always resolves. `_check` now reads *"PASS iff
C1, C3, C4 hold AND C2' resolves to a recorded branch"*, and C2' always resolves
— G, G-adv or U.

Four reasons I judge it sound rather than a loosening, each checkable:

- **It was ordered, dated and in the open**: the Review FULL's 2026-09-06
  disposition on `lt01-c2-body-cannot-rise`, not a builder's own idea.
- **The measurement is recorded**: attempt 1 (FAIL, 08-31) died on a clause
  that demanded, as a *precondition of the claim*, `nonladder_rise >= 0.6 m` on
  a body measured at **0.084 ± 0.067 m** — i.e. it demanded an observation whose
  absence is the claim's own title, and it inferred gameability from a *random*
  agent.
- **The bar did not move**: 0.6 m in both branches, `HANG_BAND`, the h(t)
  conjunction, C1, C3 and C4 all byte-identical.
- **Attempt 1's data does not retro-PASS.** The new arm is mandatory: with no
  `adv_finite` metric, V1 VOIDs. And Branch U is gated by a **new** V5 that VOIDs
  an adversary which engaged the ladder (contaminated) or read below the random
  null (dead optimiser). The net effect is that the claim must now defeat a
  *deliberate gamer* with adhesion free — strictly harder than before.

Residual, for the record and not a defect: under **Branch U** the row reads PASS
while conceding that h(t)'s *necessity* is unproven. That is honest only because
the branch is written into `c2_branch` on the ledger row and into `claim_branch`.
The pilot says it is moot — the adversary reached **0.7142 m**, so **G-adv** is
the branch that will fire and the original claim is restored as a measurement.

No other hit. No seed count reduced, no `_check` gained an `or` in the loosening
direction, no control deleted or weakened, no assertion removed. `T0.28` gained a
property (16 → 18 → 19 across the window), `T0.31` 17 → 18 → 19, `ME.1/ME.3/ME.5`
raised their distractor denominators — all tightenings.

**3. Drift from the goal — the best day for `GOAL.md` in a week, and I will say
so plainly.** Last 24 h by subject: `LT.01` C2' design + implementation + pilot
(**the Ladder Test itself — `GOAL.md:31-33`, "try to climb the ladder, fall, and
learn from falling"**), `T3.09`'s registry stamp (ledger honesty), the
`commitments_uncovered` counter (the constitutional check), two armed defaults
fired, four journals, three certificate re-buys. That is **roughly a third of the
window on the ladder-and-apple sentence**, against the ~5/26 the Review measured
yesterday and the zero the 101st audit measured the day before. The converse
question is RANK 1: `heavy`, `far`, `tiring`, `worth-it` have no passing spec
because they have **no spec**, and `curiosity` (12 specs, 2 PASS, 0 runnable),
`sleep` (5, 0, 0) and `fast/slow` (8, 0, 0) remain the neglected families — all
three sitting behind `T1.08`/`T2.01`/`LC.03`.

**4. Builder alive and productive — ALIVE, 19/19 slots fired, one lost run.**
Every hourly slot from 2026-09-18T12:07 to 2026-09-19T06:07 started; 17 ended
rc=0; the two rc=1 (15:07, 16:07 yesterday) were session-limit refusals recovered
by 17:07. PASS delta over 24 h: **0** (108 → 108), with one FIRST-EVER (`HR.1`
FAIL, honestly pre-registered) and seven settle events in the wider window. No
paused loop, no credit exhaustion, no repeated identical failure. The single
defect is RANK 2, and it is not idleness — it is a lost result.

**5. Compute honesty — the accounting is correct and I checked the week key by
hand.** `Budget._week()` is `%Y-W%U` (Sunday-start, matching Kaggle's actual
reset), **not** ISO — so today, Saturday, is the last day of `2026-W37`, and the
builder's "W38 opens tomorrow" is right where an ISO reading would have called it
wrong. Current week: kaggle **productive 1.379 h, failed 0.8373 h, used 2.2163
of 30.0, remaining 27.7837 h** — all of which **expires tonight**, and
`unattributable_hours` is **0.0**, so that figure is a fact and not a floor.
Third consecutive weekly write-off (W32 8.82 h, W33 22.11 h, W37 27.78 h). There
is still no legal buyer on the board and manufacturing one would be worse; the
101st audit's ruling stands. `gpu_hours_no_verdict` **TOTAL 48.42 h**, dominated
by **`D1.0` 33.78 h across 2 attempts and 0 verdicts** — unchanged, still the
largest single block of spend with nothing on the ledger. `overruns: []` is
correct rather than broken: `2bfa84f`'s per-job mark shipped 09-18 and no job has
been charged since, so **the guard has never fired in anger** and its first real
test is still ahead.

**6. Stuck decisions — `decisions --check` EXIT 0, `ratchet ok`
(0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0
default-action-expired, 0/0 firing-diff).** Zero `MEANS-ESCALATED`. Armed
register: **`D27` due 09-20 (tomorrow), `D28` 09-21, `D29` 09-22, `D31`
09-25** — none fires today. Four soft `CONDUCT-MISFILED?` advisories (D27, D28,
D29, D31), unchanged from the 101st audit's RANK 1 and still the honest reading
of that desk. On "was an owner decision quietly acted on without being
recorded?" — **no**: `D31`'s own entry records its option (i) as
*"SHIPPED 2026-09-18 (`2bfa84f`)"* in the options table, so the pre-emption is on
the face of the entry. I looked for the failure and did not find it.

The one thing that *is* stuck and cannot be seen by that tool is RANK 1: it lives
on a **queue row**, not in the register, so no `MEANS-ESCALATED` check reaches it.

**7. Bakeoff hygiene — CLEAN.** `D20` and `D30` both fired by armed default at
~00:2x today. Both carry the required wording (*"The owner did not rule by
2026-09-18, so the pre-registered default fired"*), both name their reversal,
both record THE LOSERS, and `D30`'s entry states its own price unprompted
(*"(v) fixes nothing. It makes the next blackout visible on the day it starts
instead of on day two"*). `--firing-check WORKTREE` ran before the commit. No
decision made without a learning gate, no VOID treated as a verdict, no winner
inside a noise margin. `champions --check` **EXIT 0, ratchet ok** on all seven
classes (0/0 phantom arena, 2/3 unfalsifiable, 4/4 unwinnable, 2/2 unverified
verdicts, 3/3 trigger debt, 1/1 kindless discharge) — `LC.03`'s VOID still holds
the Learning-core seat and the World seat still holds BY VERDICT with no deciding
run named.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?** **Today, yes — narrowly, and for the first time in five days.** Not
because the scoreboard moved (108/249 for the sixth day), but because the work in
the window was about the ladder: a clause that demanded a body do something the
claim's own title says it cannot was replaced by a measurement that pits a
deliberate gamer against the metric, and the gamer **won** — 0.7142 m over a
0.6 m bar with zero ladder contacts. That result *earns* h(t) its place instead
of assuming it, which is the whole point of Stage 0. Against that: the run that
would have banked it died 48 seconds after it started and nothing noticed; four
sentences the owner wrote about what survival is supposed to teach him still have
no falsifiable claim behind them, four days after they became countable; and the
desk that decides what gets built next is 12 promises overdue. We are closer to
the apple than yesterday. We are not closer than we would be if the loop's own
report of an empty board were checked against the constitution instead of against
the index of things that already exist.

---

## FOR THE BUILDER

1. **RANK 1, and it outranks everything including the lost run: register a
   falsifiable claim for at least one of `heavy`, `far`, `tiring`, `worth-it`
   this iteration.** This is the audit script's FIRST rule and I am not permitted
   to execute it. It needs no ruling: registering is monotone, the builder has
   done it five times unaided in seven days (`HR.1`, `LG.12`, `LG.13`, `SO.10`,
   `W1.02`), and a spec that lands **BLOCKED** on a world feature that does not
   exist still closes the hole — it gains an id, enters `run blocked`, joins
   `unreachable`, and names its own blocker, which is strictly better than
   absent. **Do NOT declare `COVERS: heavy` on `PG.2`** — it is a buoyancy/density
   spec and the nomination is a keyword artifact; crediting it would be the
   `T6.03` false-citation defect, and it would lower `commitments_uncovered`
   without covering anything. `far`/`tiring`/`worth-it` have zero nominations, so
   there is no shortcut there to resist. When you register, say in the commit
   which `GOAL.md:187` word it covers and what makes it falsifiable.
2. **Re-run `LT.01` attempt 2 IN THE FOREGROUND, and treat the whole slot as its
   budget.** ~34 min / 3 seeds projected from the 09-19 pilot record. The
   06:0x attempt is dead (no process, no artifact, no commit — evidence in RANK
   2); nothing is re-invoking anybody. When it lands, read `c2_branch`, commit the
   row by name, and the `lt01-c2-body-cannot-rise` row (DUE 09-24) becomes ACTED.
   *(The Review's 06:37 sitting, committed at `442698a` while I was writing,
   ordered the same thing as its own item 1 — two desks converging on it
   independently. Take the instruction from the desk that owns steering; this
   item exists so the evidence that the 06:0x attempt died is attached to it.)*
3. **Extend the dies-with-parent lesson to name the LANE, not only the clock —
   fifth occurrence.** The 05:18 lesson's arithmetic was *satisfied* by the slot
   that then lost the run (06:09 + 34 min = 06:43, inside the deadline). What it
   does not say, and must: **a background task, a `&`, a `setsid`-less detach or
   any "I will be re-invoked when it finishes" is not a lane — once `claude -p`
   returns, `ladder.sh` closes the iteration and nothing wakes.** With the
   detached lane closed by `D20`, the only lane for local computation is the
   foreground of a slot that stays open. Add it as occurrence five with today's
   `ps`/artifact/commit evidence.
4. **`ratchets record` must not absorb a CLOCK-driven rise into a floor.** This
   family already computes the split (`ef2757c`; `review_queue_net_arrivals`
   prints `(clock +0, act -1)` on every status run) — `record` writes one scalar
   and throws it away. Make it store the components, and keep the `!! MOVED`
   banner alive for the clock share until an act pays it down. This morning 4 of
   6 points of movement on `review_queue_violations` went to baseline unbought
   (RANK 3). Reporting-only; no floor moves; do not gate anything on it.
5. **Nothing watches for the absence of a result — build the cheapest possible
   version.** An iteration that ends with an outstanding "waiting on X" and is
   never re-entered leaves a promise in `ladder.log`, no process, no artifact and
   no ledger row, and no gate anywhere goes red. It took `ps` + `ls -lat` and a
   28-minute delay to find today's. The cheap reading: at slot start, if the
   previous slot's final log line asserts a pending run and no commit followed it,
   print `LOST-RUN: <spec> claimed in flight at <ts>, no artifact, no row`. Report
   only, unfloored. (This is `D27`'s general question — *is a recorded thing ever
   read?* — wearing its sharpest instance, so if the two want to be one reader,
   say so and build one.) **Partial overlap, declared:** the Review's `442698a`
   item 2 orders `notice_exited_dispatches()` wired onto the live path rather
   than only behind `pace_gate`. That covers *dispatches that exited*; it does
   not cover today's case, which is a **local, in-session** run that was never a
   dispatch at all. Wire theirs first — it is one line and it is ordered — then
   say plainly whether it can see a foreground-lane loss, and build the rest only
   if it cannot.

## FOR THE OWNER

**1. NO-DECISION — nothing new is asked of you today, and that is deliberate.**
The 101st audit found your desk holding four items that this project's own
instrument flags as probably the desks' own paperwork (`D27`, `D28`, `D29`,
`D31`, all `costs 0 specs`, all `class: goal`, all `CONDUCT-MISFILED?`). Adding
a fifth while that stands would make the finding worse. Every item below is
reported, not asked.

**2. `GOAL.md:186-188` — NO-DECISION, and I am overturning the recommendation
that this reach you at all.** `PROGRESS.md` item 6 carried the fork to you:
*commit to falsifiable claims for `heavy`/`far`/`tiring`/`worth-it`, or correct
`GOAL.md`'s sentence.* **Only the second arm is yours.** Registering claims is
already permitted, monotone, and something the builder has done five times
unaided in the last week; it is now item 1 of `FOR THE BUILDER`. Your sentence
stands untouched and unnarrowed, which is the safe direction. **You will hear
about this again only if registration proves genuinely impossible** — i.e. if
some primitive cannot be given a falsifiable claim in any world we can build —
and then the ask will be the honest one, with the attempt attached rather than
the fork.

**3. NO-DECISION — where your GPU ration actually went this week.** Kaggle W37
(`%Y-W%U`, ending tonight): **1.379 h productive, 0.8373 h failed, 27.7837 h
expiring unspent**, with `unattributable = 0.0` so that is a fact, not a floor.
Third week running (W32 8.82 h, W33 22.11 h). There was no legal buyer on the
board and inventing one would have been worse than the loss — I agree with the
100th and 101st audits on that and am not re-opening it. The fact worth your eye
is the *composition*: `gpu_hours_no_verdict` totals **48.42 h**, of which
**33.78 h is `D1.0` across two attempts and zero verdicts.** More hours have been
spent on one undecided arena than the entire quota we are about to let expire.

**4. NO-DECISION — the Review is executing `D28`'s default two days before it
fires, and I want that on the record in your favour.** At 06:41–06:43 today the
desk disposed two overdue rows with written designs, re-dated them in the open
with reasons and dates taken from the instrument's own `next_free_due`, recorded
the counters it moved, rewrote the steering block it owed, and named what it did
not do. That is the first reduction of `review_queue_violations` since the
thirteen broke together on 09-13. Drain is still `UNBOUNDED` and 12 promises are
still red, so nothing is fixed — but the direction reversed under its own power,
without your ruling, which is what `D28` was betting on.

**5. NO-DECISION — one thing to check tomorrow rather than decide today.** The
desk has staked a public attempt on Sunday's FULL run: *dispose the overdue class
first and in bulk, ruling rather than re-dating, before Part 2 opens a single
certificate.* `D27`'s own evidence is that the FULL sitting **has died at max
turns on four of four Sundays.** The remedy is scheduled into the least reliable
sitting this system has. If it dies a fifth time, the question stops being the
backlog and becomes the cadence — and `PROGRESS.md` item 7 already says that one
would be yours.
