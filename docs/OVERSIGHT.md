# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Running history lives in git.

**2026-09-12 06:37–07:0x UTC — 89th audit.** Window: the last 24 hours
(2026-09-11 06:51 → 2026-09-12 06:5x).

## VERDICT: DRIFTING

**The ledger is sound, nothing was loosened, no commitment is uncovered, every
ratchet is at its floor and the queue has 0 violations.** Sections 1, 2, 6 and 7
have no findings and I say so plainly below rather than dressing them up.

**The finding is that this desk's own token spend is now the marginal cause of
the builder's blackout, and I can put a number on it from a file this project
already writes.** The pace line stands 3 points below the meter. The two
document organs drew **4 points** during the 94 dark slots. Had they not run at
all, the meter would read ≤ 71 against a line of 72 and **`pace_gate` would have
released the builder at the 06:07 slot this morning.** It did not. The Review and
this audit are spending, right now, the exact headroom the builder needs — and
the builder is the **only** organ `pace_gate` applies to.

This page is deliberately shorter than yesterday's 585 lines. That is not a
style choice; it is the finding applied to its author.

---

## RANK 1 — the builder draws a quarter of this project's meter and is the only organ rationed by it; the two desks that are not rationed hold it dark today

### The arithmetic of the gate, exactly, so nothing here is a forecast

`scripts/lib_usage.sh:70-97`, verbatim:

```
PACE_FLOOR=25   PACE_CAP=90
allow = PACE_FLOOR + ceil((PACE_CAP - PACE_FLOOR) * elapsed / 100)
skip iff  pct >= allow          # pct = week:all models, elapsed = % of week
```

Live at 2026-09-12T06:38 UTC: `pct` **75**, `elapsed` **72**, `allow` **72**,
75 ≥ 72 → skip. Week resets **2026-09-14 05:00 UTC**; `week:Fable` **100%**.
**94 consecutive paced skips** — every slot since the 08:07 iteration ended
`rc=0` at 2026-09-08T08:23 (15 on 09-08, 24 each on 09-09/10/11, 7 today: the
count is exact, not estimated).

### Who actually spent the week, from `/data/jack-logs/usage_ledger.jsonl`

This is the file `D26` option (i) proposes to read, and it has been written at
every organ's start and end since `D15`'s default fired. Attribution below is the
**union** of organ intervals (the overseer and the Review overlap daily; summing
their sessions double-counts), meter rise measured between the samples bounding
each interval. Usage week, reset 2026-09-07T05:07, meter 0 → 75:

| drawn while … | points | of 75 | organ-time |
|---|---|---|---|
| the **builder** was running | **+18** | 24% | 3.8 h |
| a **desk** was running (overseer / Review / field watch) | **+7** | 9% | 2.6 h |
| **nothing of this project was running** | **+50** | **67%** | — |

The four largest idle draws, in order: **+28 points in 22.2 h**
(09-08 08:23 → 09-09 06:37), +7 in 23.8 h, +5 in **0.8 h**, +3 in 23.6 h.

**`ladder.log` resolves the onset finer than the ledger can.** The builder's own
end record reads **31%** at 08:23 on 09-08. The first paced skip, 44 minutes
later, reads **39%**. Another +9 arrived in the hour after that. An external
consumer drew roughly **+17 points in under two hours** immediately after the
builder's last completed iteration, and that single burst is what put the meter
over the line. Nothing this project did caused the blackout.

### The marginal fact, which is the one that indicts this desk

Across the 94 dark slots the meter went **31 → 75 = +44 points**. Of those,
**+4** are the two desks' own recorded sessions (09-09 +2, 09-10 +1, 09-11 +1)
and **+40 (91%)** are external. So the desks are not the cause of the outage.

**They are the cause of its last nine hours.** The gap this morning is 3 points.
Four points of desk spend sit inside it:

```
  measured now      pct 75   allow 72   gap 3     -> SKIP
  desks' draw during the blackout                    -4
  counterfactual    pct 71   allow 72             -> PROCEED at 06:07 today
```

And today's two sessions — the Review's and this one, both stamped `start` at
06:37 with `pct 75` — will add roughly **+1 more**, which under the flat-meter
condition moves the crossing from the **15:07** slot to about the **18:07** slot.
**Three more hours of dark, bought by two desks writing about the dark.**

*Stated honestly:* the meter is integer-rounded, so each session delta carries
±1; detached CPU/GPU runs do not touch this meter; and `+50` idle is
"unattributable to any organ of ours", not "provably the owner's". None of that
moves the counterfactual, which needs only the live reading and three recorded
session deltas.

### What this is not

It is **not** an argument that the audit should stop, and it is **not** the 87th
audit's recommendation returning by the back door — that report told the owner to
write `.usage-resumed` and suspend pacing, on a forecast RANK 3 shows was
impossible. Suspending the 90% stop is still the wrong act. The repair is
`D26`'s own armed default, which is **overdue and unfired** (RANK 2), plus the
discipline this page is practising: fewer lines.

**The structural asymmetry, which no instrument prints:** `scripts/review.sh:30`
calls `usage_gate` and **not** `pace_gate`. `scripts/overseer.sh:79` calls
`pace_gate` but exempts one audit per UTC day (`D15` clause (c)). So the organ
that draws 24% of the meter is rationed against a total that is 67% not its own,
and the two organs that draw 9% are rationed by nothing but their authors' care.
That sentence is the whole of `D26`.

---

## RANK 2 — five armed defaults are overdue, not three; one of them was firable by this desk and three consecutive audits routed it to an organ that has been dark for four days. I have fired it.

`decisions --check` is **EXIT 0**, ratchet at floor (0/10 undeclared, 0/3
unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0 default-action-expired) — and
prints **five** `OVERDUE — DEFAULT IS DUE TO FIRE`:

```
  D22   red 2026-09-09   4 d   default writes NOTHING        <- FIRED by this audit
  D18   red 2026-09-10   3 d   default is builder code
  D26   red 2026-09-11   2 d   default is builder code
  D23   red 2026-09-12   NEW   default is builder code   (decide_by 09-11)
  D24   red 2026-09-12   NEW   default edits CHAMPIONS.md (decide_by 09-11)
```

`D23` and `D24` went red at midnight. Yesterday's Review names three
(`PROGRESS.md:377`, `FOR THE BUILDER` item 2) and the 88th audit named three
(`OVERSIGHT.md` RANK 2, now overwritten) — **both were correct when written, and
the builder's steering now undercounts its own overdue docket by two.** The
queue has grown by one every day for four days and nothing has drained it.

### Why this is a mechanism failure and not a backlog

The armed default exists because `D1` sat OPEN for twenty days blocking 38 specs.
It replaced owner deadlock with a clock. **But every one of the twenty defaults
this project has ever fired is stamped `(builder)`** — `D1`–`D14` on 09-01,
`D15`/`D16`/`D21` on 09-06, `D17` on 09-08. The clock has exactly one executor,
and that executor is switched off. The mechanism that was built to survive
silence does not survive its own loop being paced.

Four of the five need code or a file this desk may not touch: `D18`
(`lib_procwatch.sh` + `run_spec`), `D26` (`pace_gate`'s print), `D23`
(`decisions.py`'s counter), `D24` (a `CHAMPIONS.md` label). Those are the
builder's, routed as B1.

### `D22` — fired, 2026-09-12, by the overseer

`decide_by: 2026-09-08`, red since 09-09T00:00, default **(i) THE RULE STANDS**,
whose own text reads *"Reversal: none needed — the default writes nothing."*
Recorded as an addendum to `docs/DECISIONS_NEEDED.md` with the required wording.
`decisions --check` drops it from the open set by `_SETTLED` (`decisions.py:319`,
366), exactly as `D17`'s addendum did.

**I am overruling the 88th audit's reason for not firing it, and I want that
visible.** That audit declined on two grounds: *"`D13` records that the overseer
may not edit its own script"* and *"every armed default in this file's history is
stamped (builder)"*. Neither survives contact with `D22`. `D13`'s parenthetical
in `DECISIONS_RESOLVED.md:459` is specifically about **`scripts/overseer.sh`** —
`D22` requires no script, no code and no file outside my brief. And a
consistent stamp is a convention, not a rule; three audits deep it had become the
reason the thing could not be done. My brief's clause on `OVERDUE` says *"Fire
the default … Do not silently extend the deadline; a deadline that moves when it
is reached is the deadlock it replaced."* Four days is not a moved deadline, it
is an ignored one.

**The tension in my own permissions, declared rather than hidden:** the brief
also says MAY NOT *"resolve an owner decision"*. I read firing a pre-registered
armed default as the opposite of resolving one — it executes an
already-permitted action the project armed in advance, changes nothing, and the
owner may rule (ii) or (iii) at any later date at no cost. If the owner reads it
the other way, the reversal is one `git revert` of one append and `D22` returns
to the open set unchanged. Nothing downstream depends on it.

The completion record — the `DECISIONS_RESOLVED.md` entry and the
`LOOP_JOURNAL.md` line that `D17`'s firing also carried — is **not** mine to
write and is routed as B1's first sub-item. I have not pretended the record is
complete.

---

## RANK 3 — "`pace_gate` never releases the builder" was not a mis-sampled rate; it was arithmetically impossible, from a formula both organs had already quoted

The 88th audit's lesson (`b945a06`) diagnosed 09-10's twin forecasts — the
Review's *"~6 days; `pace_gate` never releases the builder at all"* and this
desk's *"49 days … not under ANY measured rate"* — as a difference of two rates
whose sampling spread is six times its magnitude. **That is true and it is the
smaller half.** The larger half is that the *sign* of the answer was fixed by a
boundary condition in code both reports had pasted into their own text:

```
  allow(elapsed=98)  = 25 + ceil(63.7) = 89
  allow(elapsed=99)  = 25 + ceil(64.4) = 90
  allow(elapsed=100) = 25 + ceil(65.0) = 90   == usage_gate's hard stop
```

The line **converges on 90**, and `lib_usage.sh:82-84` says so in a comment
explaining why the division rounds up: *"a pace line must converge ON the limit,
not beside it."* At the end of every week `allow` equals the 90% stop, so
`pace_gate` cannot refuse a slot that `usage_gate` would allow. **The final two
or three slots of any week are released unless the hard stop fires.** "Never
releases" is not a bad extrapolation — it is refuted by evaluating the quoted
expression at its endpoint, which neither desk did, and which needs no meter data
at all.

Appended to `docs/LESSONS.md` as a correction beside the 88th's own lesson: when
you are about to divide a remainder by a rate, first evaluate the closed-form
term at its boundary — the answer is often there and carries no variance.

### Today's release condition, as a condition and not a date

Per the 88th audit's B2 (which I keep), no point forecast:

```
  now              pct 75   elapsed 72   allow 72   gap 3
  crossing needs   allow > 75  ->  ceil(0.65*elapsed) >= 51  ->  elapsed >= 77
  IF the meter holds at 75      first qualifying slot: 15:07 today
  at +1 more point (today's desks)                    ~18:07 today
  at +3 (yesterday's measured external draw)          ~22:48 today
  the meter's last four daily rises, individually     +28  +7  +3  +2
```

**And the bound that needs no assumption at all:** unless `week:all models`
reaches 90% and `usage_gate` stops everything, the builder is released before the
week resets at 09-14T05:00, because the line ends where the stop is. The Review's
published *"09-12T08:40"* is superseded — at 08:40 `elapsed` is 73 and `allow` is
73, so 75 ≥ 73 still skips. Its assumption (flat meter) failed overnight by +3.
The bound was correctly labelled; the condition simply did not hold.

---

## The audit, section by section

**1. Integrity of the ledger — NO FINDINGS.** 143 rows: **108 PASS, 22 FAIL, 13
VOID**. All 108 PASS checked directly: every `commit` still resolves in git (0
vanished), 0 marked `+dirty`, 0 absent from `BY_ID`, and **every one declares a
`control`**. Two carry empty `control_metrics` — `T0.01` and `T0.10` — and both
declare *"NONE, BY DECISION (52nd audit B5)"* with the reason stated at the spec
(an ImportError and an external service's own refusal are the falsifiers; there is
no mechanism to sabotage). That is a declared absence, not a missing control.

**2. Thresholds and controls — NO FINDINGS, and the reason is worth stating.**
`git log --since="7 days ago"` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` shows **no commit since `0d57b1d`, 2026-09-08** — the tree
has been frozen for four days because the only organ that edits it is dark.
Nothing could have been loosened. This is a true clean result and also an empty
one: silent loosening is impossible during a blackout, so section 2 will be the
first thing to matter again on the builder's first slot back.

**3. Drift from the goal — drift by production, not by direction.** 11 commits in
24 h, **0 from the builder**, 0 spec-level work, 0 ledger settlements, **fifth
consecutive day at 108/245 demonstrated**, net demonstrated 0. All 11 are
document organs correcting documents. Those corrections do serve `GOAL.md`'s
fourth clause — *"protects the honesty of watching what happens when the three
meet"* — and I am not calling them drift. But nothing in 24 hours built the
brain, the body or the world, and that is the other three clauses.

The converse question, which is the harder one. `coverage` **EXIT 2** on
`claim_dead = 4`: **smell, balance, shelter/building, thermal (kills)** have no
passing claim and every claim spec parked or foreclosed. **Two of those four —
"too cold kills him" and "he builds a shelter" — are the owner's own words and
two of the four commitments the 2026-08-10 miss was about.** They were uncovered
then by absence; they are claim-dead now by foreclosure, which is a different
disease with the same reading on the page. Nine more commitments have live claim
specs and nothing passing, and the three the brief names as most likely to be
quietly neglected are exactly where the thinness is: **curiosity 2 PASS of 12,
one brain / unison 1 of 27, learning-by-living** — `sleep` 0 of 5, `fast/slow` 0
of 8, `hunger/thirst` 0 of 6, `death & retry` 0 of 6. `0 commitments with NO
declared spec`, so the highest-priority coverage class is clean; the ladder is
the right ladder and almost none of it is climbed.

**4. Is the builder alive and productive?** Alive, firing, and producing nothing:
**94 of 94 slots fired and refused.** 0 iterations in 24 h, 0 `rc=0`, PASS delta
0. Not a crash, not credit exhaustion, not a paused loop — `pace_gate`, per RANK
1. The last real iteration (2026-09-08T08:07–08:23, `rc=0`, 108 → 108) ran on
**model `fable`** at `week:Fable` 61%; that meter now reads **100%** against
`MODEL_FLOOR=95`, so the builder will wake on **Opus**, billed to the same shared
meter it is paced against, and nothing records that substitution where a later
audit can find it. Carried from the 88th audit's B3, now asked by four audits and
the Review.

**5. Compute honesty — the finding is old, the number is worse.** `2026-W36`
(the live GPU week, Sun 09-06 → Sat 09-12 under `gpu.py::_week`'s `%Y-W%U`):
**17.7238 h charged of `KAGGLE_WEEKLY_HOURS=30.0`, 12.2762 h remaining, expiring
at the end of today.** `overruns: []`. All 8 W36 jobs resolve to ledger rows, so
nothing is unattributed this week — but **four of them (17.61 h) are `D1.0`
attempt 2, and `gpu_hours_no_verdict` reads `D1.0: 33.78 h / 2 attempts / 0
verdicts`.** The entire live GPU week bought no ledger verdict, and the project's
largest single compute line has produced none across two attempts.

The remaining **12.28 h expires in ~17 hours and cannot be spent by anyone.**
Earliest builder wake is the 15:07 slot; attempt 3 measures 17.61 h and does not
fit in 12.28; and its precondition (twin-spread result on the row → successor
gate committed in a non-dispatch commit → dispatch) is not walked. **This is a
real loss and it is also the correct outcome** — the builder's own steering said
*"it does not fit and must not be squeezed"* and that sentence has now survived
two desks trying to overwrite it. `2026-W37` opens tomorrow with a fresh 30 h.
`gpu_unattributed_jobs = 21`, AT floor.

**6. Stuck decisions — NO FINDINGS beyond RANK 2.** 0 `MEANS-ESCALATED` (no fork
a measurement could settle is on the owner's desk), 0 `UNDECLARED` (nothing to
arm — the brief's "arm at least one per audit" has nothing to bite on, which is
the right kind of empty), 0 `UNROUTED-OWNER-ASK`, 0 `VANISHED-OWNER-ASK`. I
checked the converse too: no owner decision has been quietly acted on. `D26`'s
two urgency premises were both withdrawn in the open yesterday by their own
authors, with the recommendation left standing — that is the process working.

**7. Bakeoff hygiene — one defect, already found, recorded and remediated; I
re-verified it independently and it holds.** `docs/DECISIONS_RESOLVED.md` carries
20 entries. The one soft spot is `PL.00/RENDER`: `ELIGIBLE_RANKING`
(`pl00_render_bakeoff.py:135`) selected `coarse-shadow512` at **8.594** over
`coarse-flat` at **11.483**, and `git log --diff-filter=A` returns exactly one
commit for that file — `b7324ba`, the commit that also carries the artifact. I
searched the whole tree at `b7324ba~1` for the arm names and the tie-break: **no
prior commit pins either.** So "pre-declared" had no witness. The 84th audit's B4
found this on 09-08, both live uses of the adjective were back-filled to *"declared
at `b7324ba`"* — a checkable pointer instead of an adjective — and the procedural
rule is written into `LESSONS.md`. Nothing is retracted and nothing should be:
the choice ran **against** its author's interest, which is the best evidence
available that the ranking was not fitted. No decision was made without a
learning gate; the one probe-class exception declares in its own record why the
3-sigma gate has no referent (the arms are loop configurations, not learners). No
VOID is treated as a verdict. No winner sits inside its noise margin.

**Ratchets, said out loud as `status` requires.** At floor and unchanged:
`unreachable` 93, `fail_unowned` 0, `claim_dead` 4, `goal_unrunnable` 7,
`park_release_pairs` 3, `champions_trigger_debt` 3, `gpu_unattributed_jobs` 21,
`review_queue_violations` 0, `cpu_foreclosed_now` 0. `champions --check` EXIT 0
with every class at floor, and **`ARENA-MISSING` is 0/0** — the 8 phantom-arena
seats my own brief still describes as *"8 seats today"* were repaired by
registration, the correct direction. Moved and declared: **`review_queue_net_arrivals`
4 (−25 since 09-08)**, `review_queue_piled_on` **8 (+2)**. Neither has been
re-recorded with `run ratchets record` in four days, so both have printed `!! MOVED`
every morning since — the 88th and 87th audits and two Reviews each declared them
in prose and none could record them, because recording is a repo write in the
committing organ's slot. Minor, and it is the fourth day of a true alarm firing on
a schedule, which `LESSONS.md` already names as its own failure mode.

**`review-queue`: 0 violations, EXIT 0**, re-run after the Review's concurrent
commits (`ee122e2`, `3eb3648`, `2f8c6e7` at 06:46–06:47). 46 routed — 20 OPEN, 3
HELD, 15 DISPOSITIONED, 8 ACTED. Three rows were dated **today** at 06:38
(`hr5-fixture-refuted`, `lg03-blind-twin-cannot-prove-itself-alive`,
`sm03-heldout-split-saturated`); by 06:47 **none remains dated 09-12** — the desk
cleared its own docket while I was measuring it, which is the schedule half and
the work half agreeing for once. What does not agree: **drain UNBOUNDED** (10
arrived vs 6 disposed over 7 cycles), oldest live **19 d**, and **09-13 still
carries 14 rows against a measured capacity of 6.** That day is now also the
Sunday FULL, the anatomy and completeness audits, and W37's opening quota.

**A live cross-organ write race, again, and handled.** The Review committed
`REVIEW_QUEUE.md` at 06:47:39 while this audit was running. My commits below use
`git commit --only <paths>` — the repair the 88th audit routed as B6 after
`57f67e6` swept its staged files. Worth noting: that B6 is itself still unfired
because the builder is dark, so the repair for the race is being applied by hand
by the organ that found it.

---

## 8. The honest summary

**No.** We are not closer to a curious humanoid that climbs a ladder than we were
yesterday, and for the fifth consecutive day we are not closer to a longer list
of green ticks either — 108/245, net 0, zero settlements, zero dispatches, a
frozen tree.

What changed in 24 hours is that the blackout stopped being weather and started
being something this project does to itself. The +28-point burst that began it
was external and nobody's fault. The four days since are structural: the one
organ that can move the ladder is the only one holding a ration card, the two
organs that write to the owner are holding none, and the three points that
separate the meter from the line this morning are smaller than the four points
those two organs spent writing about the blackout. **This desk did not merely
fail to catch that; it is the last nine hours of it.** Yesterday's finding was
that our errors had migrated out of the instruments and into our prose. Today's
is one step worse — the prose is no longer only inaccurate, it is *expensive*,
and it is being paid for out of the one budget the creature needs.

The encouraging facts, and they are real. Every hard number held: 108 PASS rows
verified row by row, no vanished commit, no dirty certificate, no threshold moved
in any direction, no control weakened, no commitment uncovered, no owner ask
vanished, no queue violation, every ratchet at its floor, and a desk that cleared
three dated rows this morning before I could report them as at risk. The builder's
one-sentence judgement about W36 has now survived two desks trying to overwrite it
and is the most reliable piece of prose in the repository — which is exactly what
`LESSONS.md` concluded yesterday and is worth saying twice. And one overdue
default that had been routed three times finally fired, from the desk that could
fire it, four days late.

The week's most important step toward Jack is still the one taken before this
window opened. `D1.0`'s twin-spread probe — forward passes only, both branches
pre-registered, frees 35 specs and blocks 38, the largest single unblock in the
project — is untouched for a fifth day. It needs no GPU quota and no owner
ruling. It needs one awake slot.

---

## FOR THE BUILDER

Your first act on waking is still a measurement: one line in
`docs/LOOP_JOURNAL.md` recording consecutive slots skipped (**94**), the
`week:all models` reading that released you, **and which model actually ran**.
`week:Fable` is at 100% against `MODEL_FLOOR=95`, so you will wake on Opus,
billed to the shared meter, and nothing records that substitution today. Asked by
four consecutive audits and the Review. **Then, in order — do B1 first and alone
if that is all you get.**

**B1 — Fire four armed defaults, and complete the record of a fifth.** The queue
is **five**, not the three your steering names; `D23` and `D24` went red at
midnight. None of these spends compute and none moves a threshold.
   - **`D22` — already fired by this audit.** Do not re-fire it. What is missing
     is the completion record `D17`'s firing carried: the
     `docs/DECISIONS_RESOLVED.md` entry and the `docs/LOOP_JOURNAL.md` line.
     Copy `D17`'s shape (`DECISIONS_RESOLVED.md:748`).
   - **`D18`** (red 09-10). Default **MEASURE AND REPORT, GATE NOTHING, RELAX
     NOTHING**: `lib_procwatch.sh` reads `/proc/PID/status:VmHWM` while walking
     pids it already resolves and **NAMES** any project python over the ceiling
     (name, never kill); `run_spec` records `peak_rss_mb` from
     `resource.getrusage(RUSAGE_CHILDREN)`. The ~1.5 GB figure in `SYSTEM.md`
     **stands verbatim** — not raised, not narrowed, not annotated. Journal: *"the
     owner did not rule by 2026-09-09, so the pre-registered default fired."*
   - **`D26`** (red 09-11). Default **(iv) MEASURE ONLY, GATE NOTHING, RELAX
     NOTHING** — and RANK 1 is the evidence that it is the right instrument, so
     build it to print what RANK 1 measured by hand. `pace_gate`'s skip line
     gains, beside the shared total: **this project's own attributed spend**
     summed from `usage_ledger.jsonl`'s existing start/end pairs, **split builder
     vs desks** (the union of intervals, not the sum of sessions — the overseer
     and Review overlap daily and summing double-counts), and **consecutive dark
     slots** as a ratcheted metric. Gate nothing, change no behaviour. No spec
     declares `lib_usage.sh` in `IMPL_DEPS`, so no certificate stales. Journal:
     *"the owner did not rule by 2026-09-10, so the pre-registered default fired."*
   - **`D23`** (red 09-12, `decide_by` 09-11). Default **(iii) MEASURE THE
     COMPOSITION, GATE NOTHING, TIGHTEN NOTHING**: `FAIL-UNOWNED` keeps its
     definition and its floor of 0; add the counter `FAIL-OWNED-BUT-UNDRAINED`
     beside it, from data both tools already hold. Journal: *"the owner did not
     rule by 2026-09-11, so the pre-registered default fired."*
   - **`D24`** (red 09-12, `decide_by` 09-11). Default **(iii) DECLARE, DO NOT
     DECIDE**: mark the Learning-core seat's arena `VENUE-UNAFFORDABLE` in
     `docs/CHAMPIONS.md` with the 526 h / 30 h-per-week arithmetic beside it. The
     10x scale ratio is **not** re-read and no threshold moves — one label, so
     `champions` prints the uncontestedness it already implies. Journal: *"the
     owner did not rule by 2026-09-11, so the pre-registered default fired."*

**B2 — The endpoint check, in the instrument, one line.** Per RANK 3: when the
pacing print emits the flat-meter bound, also emit **`allow(elapsed=100)`** — the
constant 90 — beside the 90% stop, so the next reader of that line can see in one
glance that the line converges on the stop and that "never releases" is
unavailable as a sentence. This replaces nothing; the 88th audit's B2 (print the
bound with its assumption, print the last N daily rises individually, print no
projected release date) stands unchanged and is still unbuilt.

**B3 — `git commit --only`, carried from the 88th audit's B6 and now twice
observed.** `git add <path>` is path-scoped; `git commit` commits the whole index,
including anything a concurrently-running organ has staged. `57f67e6` swept 280
lines of this desk's work under a message about queue rows. The Review committed
`REVIEW_QUEUE.md` at 06:47:39 today while this audit was running. Every organ
script that commits should use `git commit --only <paths>`. This is the benign
half of the live row `cross-organ-doc-race-voids-certificates`.

**B4 — `D1.0`'s twin-spread probe, and nothing ahead of it once B1 is done.**
Fifth day untouched. Forward passes only, both branches pre-registered on the
row, no GPU quota needed, no owner ruling pending: frees 35 / blocks 38, the
largest single unblock in the project. **Do not squeeze attempt 3 into today.**
`W36` holds 12.28 h against attempt 2's measured 17.61 h — your own sentence, now
twice vindicated — and `W37` opens tomorrow with a fresh 30 h. The precondition
binds under every branch: twin-spread result on the row → successor gate
committed in a **non-dispatch** commit → then a dispatch. An unchanged
re-dispatch stays forbidden.

**B5 — `PL.02`'s eye gate is RULED; implement it** (`5e39771`, DUE 09-14).
Rebind the VOID condition to a raw-pixel radius ridge R² ≥ 0.80
(`EYE_RADIUS_R2_MIN` unmoved) measured on the run's **own** probe episodes, not
inherited from the seed-90 probe. Keep `r2_ua` a first-class recorded metric on
the ledger row. Then a smoke; the registered run stays blocked until it PASSES.

**B6 — `run ratchets record` for the two `review_queue` counters**, in whichever
commit justifies them. `net_arrivals` 4 (−25) and `piled_on` 8 (+2) have printed
`!! MOVED` every morning for four days; four organ-runs have declared them in
prose and none could record them. Also: **`W1.04` still gains conjunct (c) before
you register it** — register from the amended design, not the 09-06 text.

---

## FOR THE OWNER

**1. `D26` is the entry to read, and RANK 1 is the measurement it was waiting
for. My recommendation is unchanged: (i) ATTRIBUTE THE LINE.** From your own
`usage_ledger.jsonl`, this usage week: the builder drew **18 of 75** meter points
(24%) in 3.8 h of work; the overseer and Review together drew **7** (9%); **50
(67%) were drawn while no organ of this project was running at all**, including
**+28 in a single 22-hour window** that is precisely what switched the builder
off. `pace_gate` rations the builder against a total that is two-thirds not its
own, and `scripts/review.sh:30` calls `usage_gate` **without** `pace_gate`, so the
two organs that report to you are rationed by nothing. **(i) does not touch the
90% hard stop** — `usage_gate` keeps reading `week:all models` unchanged, so a
genuinely exhausted pool still stops everything.

**I am not asking you to hurry, and I am not asking for an override.** The 87th
audit handed you a `.usage-resumed` command on a forecast that RANK 3 shows was
impossible from the code it quoted; that recommendation is withdrawn and stays
withdrawn. On today's arithmetic the builder wakes this afternoon regardless, and
the line ends where the stop is, so it wakes before the week resets under any
draw short of the hard stop. **What (i) buys is not this afternoon — it is the
next time somebody else uses the account.** The overdue default **(iv) MEASURE
ONLY is unaffected by any of this and should still fire**; it is routed as B1.

**2. NO-DECISION, but you should know it: I fired `D22`'s armed default this
morning, and I am the first non-builder organ to fire one.** `decide_by`
2026-09-08, red four days, default **(i) THE RULE STANDS** — design authority
stays with the Review, unchanged and unnarrowed, *"the default writes nothing."*
The owner did not rule by 2026-09-08, so the pre-registered default fired.
Recorded in `DECISIONS_NEEDED.md` with the required wording. **Reversal: revert
one append; `D22` returns to the open set with its options, default and
`decide_by` untouched, and you may rule (ii) or (iii) at any later date at no
cost.** Two audits before me declined to fire it on grounds I judge do not apply
(RANK 2 states them and my reasons in full); my brief's MAY-NOT list says I may
not *resolve* an owner decision and I read executing a pre-armed default as the
opposite of resolving one. If you read it differently, the revert costs nothing
and I would rather be told than guess.

**3. `D18`, `D26`, `D23` and `D24` remain OVERDUE and unfired**, repeated
verbatim rather than dropped because a `VANISHED-OWNER-ASK` is the scar this
section exists to prevent. The queue has grown by one a day for four days. Every
one of the four defaults is a measure-only or declare-only act that moves no
threshold and refuses no run, and **every one needs the builder**, because the
armed-default mechanism has exactly one executor and no clause for that executor
being switched off. That is the finding under RANK 2 and it is yours to know
rather than to rule on.

**4. NO-DECISION: 12.28 free GPU-hours expire tonight and cannot be spent.**
`2026-W36` charged 17.72 h of 30; the remainder dies at the end of Saturday. No
organ can spend it: the builder's earliest wake is this afternoon, attempt 3
measures 17.61 h and does not fit, and its precondition is unwalked. **This is a
loss and also the correct outcome** — squeezing it is the act two desks spent
three days wrongly telling your builder to perform, and its own steering refused.
`2026-W37` opens tomorrow with a fresh 30 h. The larger number behind it:
`D1.0` has now consumed **33.78 GPU-hours across two attempts for zero ledger
verdicts**, which is the project's single largest compute line.

**5. NO-DECISION: five days at 108/245, and the ladder is the right ladder.**
`coverage` reports **0 commitments with no declared spec** — the 2026-08-10 hole
is closed. But four of your constitutional commitments are **claim-dead**, two of
them in your own words: *"too cold kills him"* and *"he builds a shelter"*. Every
claim spec behind them is parked or foreclosed on honest, evidence-backed
verdicts, so nothing was hidden and nothing should be deleted — the repair is a
registered successor spec, and it is routed
(`five-commitments-are-claim-dead-behind-foreclosures`, DUE 09-16). `coverage`
keeps exiting `rc=2` until it lands, correctly.
