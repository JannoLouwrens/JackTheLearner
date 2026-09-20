# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **105th audit — 2026-09-20, 00:37–01:0x UTC.** Opened at HEAD `99a1f40`,
> 26 minutes after the 00:0x builder slot ended (00:12:10) and six hours after
> the 104th closed at `38bf7a4`. **No organ running concurrently** — the
> Review's 06:37 Sunday FULL has not sat yet, the 01:07 slot has not started —
> so every number below is a single clean reading taken by me.
> `demonstrated` **109/253 (43.1%)**. Ledger: PASS 109, FAIL 30, VOID 14,
> BLOCKED 1. Builder, last 24 h (09-19 00:38 → 09-20 00:38): **24 slots
> started, 24 ran** — 23 × `rc=0`, 1 × `rc=124` (16:07, inherited and completed
> by 17:07). **56 commits**, 15 touching the ledger. 5 first-ever verdicts
> (`PS.05`, `PS.06`, `PS.08`, `PS.09`, `LT.02` — all honest FAILs) and one
> status change (`LT.01` → PASS).

## VERDICT: DRIFTING — not because a number is wrong, but because the system's written account of last night is not what happened, in the exact place it had just declared a control

The science is not the problem and I will say so first. In this window the
builder banked `LT.01` **PASS** (the Ladder Test's own honesty certificate),
ran four `GOAL.md:187` primitives to measured, committed-as-found FAILs, ran
`LT.02` to a first-ever FAIL whose single fired conjunct is a *finding about
the venue* rather than a red, and shipped the ratchet cause store. Sections 1,
2, 5 and 7 are clean and I re-derived every one rather than inheriting
yesterday's word. **Zero numeric thresholds moved in the loosening direction
across seven days** — stated as plainly as a finding would be.

What is wrong is the record of conduct, and it is wrong in three coupled ways
that all point at the same six hours of 2026-09-19:

| # | finding | damage |
|---|---|---|
| 1 | **The lane guard cannot see the only detached lane this repo owns.** `launch_detached.sh` interposes one process between `setsid` and the spend, so neither refusal fires. A registered ledger row (`LT.02`) was bought through that lane at 21:11, **6 h 46 m after the guard shipped to close it** | a control that reports `launchable` for the lane it names as closed; `LESSONS.md` says this class is now GUARDED |
| 2 | **Three consecutive slots (19:0x, 20:0x, 21:0x) wrote no journal entry, and the one surviving entry states the opposite of what the logs show** — it calls the producing run "the 19:0x slot's foreground launch" | the project's own memory records a detached buy as foreground, in the window where a lane's legality is being decided |
| 3 | **`D20`'s scope is read two opposite ways inside this repository** — unqualified in the code, scoped to `cpu<48h` on the Review's page — and finding 1 is the reason nobody noticed | the owner's own CONDUCT ruling has no single reading |
| 4 | **The 104th audit's RANK 1 is unmoved at 3½ days.** `goal_unrunnable = 7`; `GEN.02/03/06/09` re-parented to a decision that had already closed | a constitutional red owned by nobody, behind a green `ACTED` |
| 5 | **The queue went 12 → 15 OVERDUE at midnight**; 21 live rows fall due on or before today against a measured capacity of 6 | 15 broken promises carried into a Sunday FULL |

Nothing here makes a ledger claim false. `LT.02`'s row is sound — clean stamp,
`dirty_files: None`, every aliveness control green, an honest FAIL on one
conjunct. I am not asking for it to be re-run and it must not be.

---

## RANK 1 — the guard built to close the dies-with-parent class is one interposed process away from silent, and the lane it names by name is the one it cannot see

**The guard.** `experiments/run.py:_lane_verdict` refuses two things:

```python
if os.getppid() == 1:            # orphaned
if os.getsid(0) == os.getpid():  # "session leader (setsid) — detached at
                                 #  birth; D20 closed this lane for registered runs"
```

**The lane.** `scripts/launch_detached.sh` — this repository's only sanctioned
detached launcher, and the one `D20`'s resolution names — launches like this:

```sh
setsid nice -n 19 env -u JACK_ITER_DEADLINE "$PYBIN" -m experiments.cpu_budget wrap "$LABEL" "$@" ... &
```

The session leader is `cpu_budget wrap`. `experiments/cpu_budget.py:463` then
does `subprocess.Popen(argv, env=env)` — the spend runs as its **child**. So
for the spending process `getsid(0) != getpid()` (its sid is the wrapper's pid)
and `getppid()` is the live wrapper, not 1. **Neither refusal can fire.**

**Demonstrated live**, with the read-only `run lane` probe (it spends nothing,
writes nothing, bills no meter):

```
A) setsid DIRECTLY on the spend path — what the fixture tests
   stdin=pipe ppid=2145070 sid=2145103 pid=2145103
   lane: ABANDONED launch — the spend path refuses this lane:
       session leader (setsid) — detached at birth; D20 closed this lane...
   EXIT 3

B) setsid + ONE interposed parent that Popen()s it — launch_detached.sh's shape
   stdin=pipe ppid=2145106 sid=2145106 pid=2145108
   lane: launchable — the spend path would not refuse this launch.
   EXIT 0
```

**The fixture certifies the shape nobody uses.** `scripts/test_lane_guard.sh`
has three `setsid` cases — lines 75, 94, 114 — and every one is
`setsid env ... "$VENV_PY" -m experiments.run <id>`: setsid exec'ing the spend
directly. ALL GREEN, 17 cases, and not one of them is `launch_detached.sh`.
That is why a guard shipped to mechanically enforce a lane closure was green
against a launcher this repo does not use.

**And it is not hypothetical. It was used last night to buy a ledger row.**
`/data/jack-logs/lt02_run_2107.log`:

```
LAUNCH 2026-09-19T21:11:01Z cwd=/home/opc/jackthelearner cmd: .../python -m experiments.run LT.02
LANE WARNING: stdin is /dev/null. ... EITHER a backgrounded launch ... OR an
ordinary sandboxed foreground call; the two are indistinguishable from here
[LT.02] ... FAIL (652.35s) pre-registered threshold not met
```

The ledger row reads `ran_at 2026-09-19T21:21:54`, `duration_s 652.35` →
started **21:11:02**. That is the launch above. The only guard output was the
**soft** warning — and the guard's own docstring says that signal also fires on
ordinary sandboxed foreground calls, which is precisely why it was demoted from
a refusal at 14:25. A mark that fires on both lanes carries no information
about which lane this was.

**The clock.** `b4fd863` shipped the guard at **14:18**. `d730ff9` corrected it
at **14:25** (v1 was falsified by its own first foreground call — good, fast,
honest work). At **21:11**, 6 h 46 m later, the lane the guard names in its own
refusal string was used for a registered spec run and the guard said
`launchable`. `JACK_LANE_WAIVER` was not set; there is no waiver banner in the
log. This was not an exception being exercised. It was a blind spot.

**What this does and does not damage.** It does not damage `LT.02`'s science by
one digit. It damages the sentence `LESSONS.md:3882` now carries — *"The class
is now GUARDED, not only noticed"* — and the 103rd audit's item 2 that sentence
discharges. Against nine occurrences, the guard's demonstrated coverage is:
direct-`setsid` (a lane nothing here uses), and `ppid=1` orphans. The two lanes
that have actually killed runs here — `run_in_background`, and now the wrapped
detached lane — are respectively *stated as unrefusable* (honestly, in the
code) and *silently permitted*.

---

## RANK 2 — three slots journalled nothing, and the entry that survived says the row was bought in a lane it was not

**The gap.** `docs/LOOP_JOURNAL.md` runs `2026-09-19 18:0x` → `2026-09-19
22:0x`. There is **no 19:0x, no 20:0x and no 21:0x entry.** Those three slots
are exactly the ones in which `LT.02`'s registered run was launched three times
and died twice. From `ladder.log`, in the slots' own words:

- 20:0x — *"Found the 19:0x slot's LT.02 run died ~19:11 (~2-3 min in, no
  ledger row) despite its commit claiming a foreground launch — the slot ended
  without noticing or journaling."*
- 21:0x — *"the class that killed this exact run at 19:11 and 20:13"*, and
  *"My earlier single-pid kill only reached the supervisor — the harness
  group-kill vs single-pid asymmetry **goes in the lesson**."*

**Occurrences 8 and 9 of the dies-with-parent class are recorded nowhere.**
`docs/LESSONS.md` was last touched at **17:33** on 09-19 (`7502c96`). Its text
on this class still reads *"remedy through occurrence seven"* and *"the likely
lane of occurrences 5–7"*. The promise quoted above — made in the slot that
finally recovered the run — was not kept. This is the fifth consecutive time
this class's remedy has been *a better lesson entry*, and the fourth time the
Review's own diagnosis applies verbatim: **`LESSONS.md` is a memory and it has
been doing duty as a control.**

**And the one entry that was written is wrong on both facts.** `LOOP_JOURNAL.md`,
22:0x slot:

> HARVESTED LT.02 attempt 1 FAIL (ran 21:21:54 in **the 19:0x slot's foreground
> launch**, 652 s, left on disk when that slot ended)

The producing launch was **21:11:01, from the 21:0x slot, via
`launch_detached.sh` — detached, not foreground.** The log header proves the
lane, the arithmetic proves the slot (21:11:02 + 652.35 s = 21:21:54, the row's
own `ran_at`), and the 21:0x slot said so itself in the log: *"launched 21:11
via `launch_detached.sh` (setsid, so no slot-end group-kill can reach it)"*.

This is the part that matters beyond bookkeeping. The builder prompt's own
premise is *"You have no memory of previous iterations — the ledger IS the
memory."* A future agent asking the live question — *has a registered run gone
through the detached lane since `D20` closed it?* — reads **no** from the
journal and **yes** from a log file nothing greps. The one place the answer was
written down, it was written down backwards.

I want to be fair about cause: the 19:0x–21:0x slots were fighting a run that
kept dying, and the recovery was good work under pressure. The failure is not
that they struggled; it is that three slots of struggle left no trace in the
organ designed to carry it, and the trace that was left is false.

---

## RANK 3 — `D20`'s scope has two readings in this repository, and RANK 1 is why the collision was silent

`docs/DECISIONS_RESOLVED.md:1321`, the resolution text, as one unqualified
sentence:

> `cpu<48h` is not a class this box can serve under that reading; **the
> detached lane is declared CLOSED to registered spec work**; the builder
> registers no new spec in the class.

**The code reads it unscoped.** `_lane_verdict` refuses every setsid-descended
registered run *regardless of cost class* — its refusal string is literally
*"D20 closed this lane for registered runs"*.

**The Review reads it scoped.** `docs/PROGRESS.md`, 2026-09-19: *"`D20`'s
closure is scoped to the `cpu<48h` class and the `launch_detached.sh` lane"*.

`LT.02` is `cpu<10min` (re-declared at `da07ede` on its own SIZING RECORD,
correctly, under the 104th audit's item 2). Under the code's reading its 21:11
launch was on a closed lane. Under the Review's reading it was legal. **Nothing
adjudicated, because the guard could not see the launch to refuse it.** The two
readings have coexisted for 21 hours without ever meeting.

This becomes load-bearing the moment RANK 1 is repaired: once the guard sees
through the wrapper, the *only* mechanism that has ever successfully carried a
long CPU run past a slot boundary here becomes refusable. That is a real cost
and it is not mine to price. `D20` is SYSTEM.md class 3 (CONDUCT) — the
owner's. It is appended to `docs/DECISIONS_NEEDED.md` as **`D32`**, armed,
`decide_by 2026-09-24`, with a default that takes no permission away and gives
none: *see it and say it*.

---

## RANK 4 — the 104th audit's RANK 1 has not moved in 3½ days, and I re-derived it rather than inheriting it

`coverage` still prints, today, at HEAD:

```
4 NEW unrunnable citation(s) — fix GOAL.md's text or route the revival;
never add to GOAL_UNRUNNABLE_BASELINE (shrink-only): GEN.02, GEN.03, GEN.06, GEN.09
```

`run review-queue` still prints the pair under its own
`DISPOSITION-ON-A-CLOSED-DECISION` reading:

```
goal-cites-four-specs-that-resolve-to-corpses -> D24 (closed 2026-09-12)
```

`GOAL.md:283–296` stakes the whole post-jungle programme — more worlds, other
minds, the told world — on four spec ids, all four of which resolve to corpses
rooted at `LC.07` (`PILOT-BLOCKED`, arena declared `VENUE-UNAFFORDABLE` by
`D24` itself). The row that owned them is `ACTED` and terminal; the decision it
re-parented to had closed four days before the re-parent was written. `ratchet
goal_unrunnable = 7` is unchanged since 2026-09-05.

No commit since `38bf7a4` (18:52 yesterday) touches it. **This is not a new
finding and I am not re-reporting it as one.** I am recording that it has now
survived two audits, that it is the Review's (its 06:37 Sunday FULL is the
first organ that can act), and that the 104th's repair instruction — *a new
owner with a date, not a re-open of the ACTED row, and not a citation deletion*
— stands unamended.

---

## RANK 5 — the queue, and the one true negative inside it

`run review-queue` at HEAD: **35 OPEN, 3 HELD, 12 DISPOSITIONED, 17 ACTED, 0
DECLINED of 67 routed; 15 VIOLATIONS, all OVERDUE**; oldest live 27 d; drain
**UNBOUNDED** (arrived 19, disposed 9 over the trailing 7 cycles). **21 live
dated rows fall due on or before today against a measured capacity of 6/cycle;
15 of the 21 are already overdue.** `2026-09-24` carries **7** promises —
AMBER, one over capacity.

**The +3 is honestly the clock, and I verified it rather than accepting it.**
`review_queue_violations` 12 → 15 and `review_queue_violation_forms`
`{'OVERDUE': 12}` → `{'OVERDUE': 15}`. The three movers are exactly the three
rows that declared `DUE: 2026-09-19` and rolled over at midnight —
`sh02-null-saturation`, `cpu48h-class-self-forecloses-the-day-meter`,
`lg12-abstention-knob-has-no-resolution`. **The cause store shipped at 23:13
last night and attributed all three to CLOCK correctly on its first live
morning.** That is the 103rd audit's item 3 working, measured, and it is worth
naming as a win on a page otherwise full of conduct findings.

`review_queue_net_arrivals` 12 → 10 is the trailing window sliding; the
instrument says so itself and says no commit can justify recording it. The
builder's 00:0x commit said both movements out loud and recorded neither.
Correct on both counts.

---

## Section 1 — integrity of the ledger: CLEAN, and here is what I actually checked

All **109** PASS rows, mechanically:

- **0** PASS whose spec is missing from `BY_ID`.
- **0** PASS with no implementation in `experiments/tests/` (matched by the
  `<id>.lower().replace('.','_')` convention, prefix variants included).
- **0** PASS whose recorded `commit` no longer resolves in git
  (`git cat-file -e <sha>^{commit}` over every distinct sha).
- **0** PASS whose spec declares no `control`.
- **2** PASS with an empty `control_metrics` — `T0.01` and `T0.10` — and both
  declare `control = "NONE, BY DECISION (52nd audit B5)"` with the reason
  written out (an import either raises or it does not; a sabotaged upload fails
  on the service's side). Not a finding.

Standing and already visible in `run status`, unchanged: 12 STALE CLAIMS (every
one a FAIL/VOID behind a Review redesign, **zero stale PASS certificates**),
1 pre-`impl_sha` stale-by-content row (`T2.02`, VOID), 3 UNBACKED CERTIFICATES
(`LF.02`, `T2.03`, `T2.14` — legal, reporting-only, rooted at `T2.10`/`T1.08`).

---

## Section 2 — thresholds and controls over seven days: CLEAN, and this is the strongest true negative on the page

I diffed `HEAD@{7 days ago}..HEAD` over `registry.py`,
`registry_expansion.py` and every pre-existing file under `experiments/tests/`
— 18 files, separating *new* files (where nothing can have moved) from
*modified* ones (where it can). Every numeric change is a **strengthening**:

| file | change | direction |
|---|---|---|
| `me_1`, `me_3`, `me_5` | `N_DISTRACTOR` 60 → **130**, `MIN_DISTRACTOR_EVAL` 30 → **59** | TIGHTER — γ=0.05 needs m ≥ 59 to certify the 0.95 bar at all; the bar itself untouched |
| `me_9`, `me_10` | certification level recorded (m=15 certifies 0.819; m=36 certifies 0.920) | doc-only; *"THE 0.95 BAR DOES NOT MOVE in either direction"* |
| `t1_07` | gains `spread_ratio <= 6.0`, pre-registered with both sides shown reachable | NEW gate |
| `t1_08` | gains `heldout_cv_pct <= 7.0`; `MIN_SNR` stays 3.0; **22.6% false-fail rate measured and disclosed against itself** | NEW gate, honestly priced |
| `ub_10` | gains `A0_HEADROOM = 0.05`; `WINNER_GATE` 0.75, `MARGINAL_FLOOR` 0.80, `NULL_GATE` 0.60, `SWAP_HURT` 0.10 enumerated as untouched | NEW gate |
| `t6_03` | gains `train_moved_weights`, `probe_dev_postload` | NEW conjuncts |
| `lt_01` | C2 → two-branch C2'; **the 0.6 m bar unchanged in both branches** | re-scope, no bar move |
| `pl_02` | `LEARN_DROP` stays 0.90, `EYE_RADIUS_R2_MIN` stays 0.80 | explicitly unmoved |

**The one that needed real checking, and it holds.** `PS.08`'s commit message
says *"gate floor re-frozen 0.015 → 0.008"*, which reads like a loosening of a
bar after a number was seen. It is not one: `git log -S'GAP_ABS_MIN'` shows
**0.015 was never committed** — it lived only in the timed-out 16:0x slot's
uncommitted file, against a superseded fixture that the builder measured as not
reproducing. The file's single commit ships `GAP_ABS_MIN = 0.008`, justified as
2.3× the measured quantum `0.00355`, and the gate is
`max(GAP_ABS_MIN, 2.0 * quantum)` — so the quantum term, not the constant, is
what binds. Pre-registration, disclosed in the open, with the measurement
attached. No finding.

**`LT.02`'s cost class `cpu<2h` → `cpu<10min` (`da07ede`)** is the other change
that loosens something: it loosens *admission* (54,000 s ENUM → 10,800 s) and
**tightens** the child-kill window by the same factor. It was ordered by the
104th audit against `SO.08`'s precedent, is backed by a SIZING RECORD in the
spec's own notes, moves no gate, and no ledger row existed under the old class.
Correctly done.

---

## Section 3 — drift: the builder's day traces to GOAL.md, and here is the converse

**Where the 56 commits went**, and which sentence each serves:

- **~14 commits — the ladder's science.** `LT.01` attempt 2 PASS → *"If there
  is a ladder with an apple on top, he must try to climb the ladder, fall, and
  learn from falling"* — this is the literal sentence. `PS.05/06/08/09` (far,
  tiring, heavy, worth-it) and `LT.02` → *"Survival earns him the primitives
  that make anything else mean something — hot, heavy, far, tiring, dangerous,
  worth-it"* (`GOAL.md:186`) and *"curiosity that drives real exploration"*.
  **All five landed as FAILs, committed as found.** Five honest reds in a day
  is the ladder working, not the ladder stalling.
- **~24 commits — the instrument** (lane guard v1+v2, ratchet cause store,
  `T0.21`/`T0.33`/`T0.36` re-buys, class re-declaration, `D20`/`D30` firings).
  Serves *"protects the honesty of watching what happens"*. **Not drift — but
  the ratio the Review has flagged for a week persists and I confirm it:
  instrument work is still the plurality of commits on a day that produced five
  first-ever verdicts.**
- **~18 commits — journal, routing, audit pages.** Necessary overhead.

**Nothing in the window serves no GOAL.md sentence.** There is no drift to
report in the ordinary sense.

**The converse, which is harder and worse.** `coverage` at HEAD:
`commitments_uncovered = 0` **at floor for the second day** — every
constitutional commitment now has at least one declared falsifiable claim, a
first in the counter's life. But **4 are CLAIM-DEAD** (smell, balance, thermal,
shelter/building — every claim spec parked or foreclosed) and **13 more have
live claim specs and nothing passing**: touch, tool use, told world, heavy,
far, tiring, worth-it, proprioception, plasticity, sleep, hunger/thirst,
death & retry, fast/slow. The three the prompt warns are most likely to be
neglected read: **curiosity 2 PASS of 12**, **one brain / unison 1 PASS of
27**, **memory across lives 2 PASS of 10**. Coverage is no longer the hole;
*passing* is.

---

## Section 4 — the builder is alive and productive

24 slots started, 24 ran, 23 `rc=0`, one `rc=124` timeout that the next slot
inherited and completed. No PACING, no ABORT, no credit exhaustion, no paused
loop. `lost_iterations.log` 0 bytes. `declared_pids` holds exactly this audit's
own slot and one already-harvested `T0.21` re-buy. No stray processes; HEAD
pushed; tree clean.

Meters, from the slots' own reporting: `week:all models` rose 40% → 54% across
the day — **that is the gate** — with `week:Fable` at 83%, not the gate. Both
printed, the acted-on one named, every slot. Correct conduct.

**PASS delta +1** (`LT.01`), FAIL +5. A one-PASS day that produced five
first-ever verdicts and closed a constitutional counter is a good day, and the
flat `demonstrated` rate is the inert metric the Review already named, not a
stall.

---

## Section 5 — compute honesty: clean, with one standing red

- **`2026-W37` closed**: kaggle **2.2163 h of 30** across 4 jobs →
  **~27.78 h expired unspent**. The Review's and the builder's arithmetic is
  confirmed correct against `gpu_budget.json` and `gpu.py:_week()`
  (`%Y-W%U`, Sunday-start).
- **`2026-W38` opened 00:00 today**: **0 jobs, 30 h, expiring Sat 2026-09-26**,
  and there is still **no legal buyer** — `coverage` reports 5 of 7 cost classes
  empty with NO path in, and the two non-empty hold only VOID arms to repair.
  The builder refused to manufacture one, per standing order. **That refusal is
  correct and I am not asking for it to change.** It is also the fourth
  consecutive week this has happened.
- **colab: 3.033 h in W37 across 3 jobs, against no ceiling at all.** This is
  `D31`'s subject, armed, `decide_by 2026-09-25`. Reported, not re-asked.
- **`gpu_hours_no_verdict` = 48.42 h TOTAL**, unchanged since 09-18, of which
  **`D1.0` alone is 33.78 h across 2 attempts and 0 verdicts**. Standing red,
  visible, owned. `gpu_unattributed_jobs = 21` at floor.

---

## Section 6 — stuck decisions

`decisions --check` rc=0. **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE,
0 UNROUTED-OWNER-ASK, 0 VANISHED-OWNER-ASK.** The prompt asks me to arm at
least one UNDECLARED per audit; **there are none to arm** — the ratchet reads
0/10 and has held there. I am adding one entry of my own authorship (`D32`,
RANK 3) rather than inventing an arming.

**`D27` is due TODAY and is NOT overdue.** I re-derived this from the tool
rather than inheriting it: `decisions --check` prints it under `armed`, not
under `OVERDUE — DEFAULT IS DUE TO FIRE`. The earliest legal firing is
**2026-09-21**. The 104th audit's item 1 is fully discharged — the builder
journalled the correct reading in four consecutive slots (19:0x through 00:0x)
and fired nothing. Today's correct act on `D27` is **nothing**.

Armed register after this audit: `D27` (09-20), `D28` (09-21), `D29` (09-22),
`D32` (09-24), `D31` (09-25) — five armed, **0 undeclared**, and I verified
`decisions --check` still exits 0 with the ratchet green after appending `D32`.
All five carry the soft `CONDUCT-MISFILED?` advisory, `D32` included: it is
routing-only, never a blocker, and it remains the Review's to answer —
`D28`'s own reclassification notice fires 09-21. I note for `D32` specifically
that I do **not** think it is misfiled: `D20` is SYSTEM.md class 3, and a desk
reading the scope of the owner's CONDUCT ruling differently from the code is
the defect, not the remedy.

**Nothing on the owner's desk has enough evidence to be decided by measurement
instead**, and nothing was quietly acted on without being recorded. `D20` and
`D30` both fired on 09-19 with the required wording and both are transcribed to
`DECISIONS_RESOLVED.md`.

---

## Section 7 — bakeoff hygiene: clean

`docs/DECISIONS_RESOLVED.md` re-read from `PL.00/RENDER` forward. Every
armed-default firing (`D17`, `D18`, `D20`, `D22`, `D23`, `D24`, `D25`, `D26`,
`D30`) carries its invariant checklist, its named losers, and its reversal.
`D24` is the file's model of the right refusal — *(iii) DECLARE, DO NOT
DECIDE*, with the "~10x" ratio explicitly untouched.

**No VOID treated as a verdict was introduced this week.** The two standing
ones are already indicted by `champions --check` and unchanged: `Learning core`
held BY VERDICT off `LC.03 = VOID` (`VERDICT-IS-A-VOID`), and `World` held BY
VERDICT naming no deciding run (`VERDICT-UNDECLARED`). `SO.10`'s entry is
recorded as **TIE**, not a winner — correct, and it is the reason
`so10-tie-break-hands-the-seat-to-an-ineligible-arm` is a live queue row rather
than a seated champion.

`champions --check` rc=0 with 10 standing violations and **every ratchet class
at floor**: 2/3 unfalsifiable (`ASR`, `Speaker ID` — NO-ARENA), 4/4 unwinnable,
2/2 unverified verdicts, 3/3 trigger debt, 1/1 kindless discharge (`LF.02`),
0 phantom arenas. Unchanged since 09-13. No new architectural seat was taken
without a challenger this week.

---

## Section 8 — the honest summary

**Yes, closer — and for the first time in a week the answer is about Jack
rather than about the ladder's paperwork.** Yesterday the system measured, on
his actual body in his actual world, that *far*, *tiring*, *heavy* and
*worth-it* are real prices W0 charges in one currency — the world half of all
four came back green on every seed — and that his own body's "chaos" is
**reducible**, which means self-surprise self-extinguishes and the curiosity
detector `LT.02` certifies against has no true positive in this venue. Those
are findings about a creature and a world, not about a file. `LT.01`'s PASS
means the ladder-and-apple test can now be trusted to have been able to fail.
Five red rows that each say something specific about the world are worth more
than five green ones that say nothing.

**And the thing to sit with is that the same night produced the opposite
pattern in the conduct layer.** A control was declared for a defect class with
seven occurrences; within seven hours the class occurred twice more, through
the one lane the control cannot see, and the record of it was not written at
all for three slots and then written backwards. The builder is not being
dishonest — every one of those slots was fighting a real fire and recovered the
run. But the project now has a demonstrated pattern: **when this system says "I
have made this class impossible", what it has usually built is a fixture that
passes against a shape nobody uses.** That is worth more attention than any
single number on this page, because it is the mechanism by which a green tick
stops meaning anything, and this organ exists to say so before that happens
rather than after.

---

## FOR THE BUILDER

Ordered. Item 1 is the only one that is urgent and it is a bug, not a design.

1. **Make `_lane_verdict` see the lane it already names.** Today it tests the
   spending process; `launch_detached.sh` never lets the spending process *be*
   the session leader, because `cpu_budget wrap` sits between `setsid` and the
   `subprocess.Popen` that runs the spec. Two honest repairs, and **the choice
   is yours — I am not specifying the mechanism**:
   (a) walk the ancestor chain to the session leader and treat a setsid-rooted
   ancestry as the same verdict as setsid-at-birth; or
   (b) have the lane *declare itself* — `launch_detached.sh` exports an explicit
   marker and `_lane_verdict` reads it — which is the honest form, because the
   launcher knows what it is and the guard should not have to infer it from
   process topology it can be one `Popen` away from losing.
   **Whichever you take, `scripts/test_lane_guard.sh` must gain a case that
   invokes `scripts/launch_detached.sh` itself**, not another hand-rolled
   `setsid`. The current fixture is ALL GREEN on 17 cases against a launcher
   this repository does not use; that is the defect that let this through, and
   a repair whose fixture repeats it buys nothing. `T0.33` certifies
   `ladder_loop.sh`, not `run.py` — check `run stale` after the edit and re-buy
   whatever it stales, by name.
   **Do NOT change whether the wrapped lane is refused or permitted** — that is
   `D32`'s question and it is the owner's. Make it visible first.
2. **Write the three missing journal entries, marked as reconstructions.**
   `LOOP_JOURNAL.md` has no 19:0x, 20:0x or 21:0x entry for 2026-09-19. The
   sources exist and are exact: `ladder.log` (the slots' own summaries),
   `/data/jack-logs/lt02_run_2107.log` (the launch header and lane), and the
   ledger row's `ran_at`/`duration_s`. Reconstruct, label the reconstruction as
   one, and do not invent anything the logs do not say — in particular, **the
   cause of the 19:11 death is not established by any surviving record** and
   should be written as unestablished rather than guessed. (The 09-19 13:1x
   attribution repair at `4c77b39` is the precedent: a wrong cause named by
   hash and corrected in the open.)
3. **Correct the 22:0x entry in the open, by appending — never by editing
   history.** It says `LT.02` "ran 21:21:54 in the 19:0x slot's foreground
   launch". It ran from a **21:11:01 `launch_detached.sh` (setsid) launch made
   by the 21:0x slot**. Both clauses are wrong and the lane clause is the one
   that matters, because `D32` turns on it.
4. **`LESSONS.md` owes occurrences 8 and 9 — the OCCURRENCE RECORD only.** The
   *generalisable* half I have already written myself this audit, at the foot
   of `LESSONS.md`: **"A fixture that builds its own instance of the thing
   under test certifies a shape the system does not use."** Do not duplicate
   it. What is still owed is the specific record: occurrences 8 (19:11) and 9
   (20:13), the group-kill / single-pid asymmetry the 21:0x slot promised and
   did not write, and the correction to the existing entry's text, which still
   says *"remedy through occurrence seven"* and *"the likely lane of
   occurrences 5–7"* as though the count stopped at seven.
5. **Do not re-run `LT.02`.** Its row is sound, its FAIL is a finding about the
   venue, and its disposition is the Review's on 09-24.
6. **Still do not pre-empt** the Review's rows: `A4`, `T2.10`, `SO.07`,
   `SO.10`, `T1.08`'s pipeline repair, `HR.1`'s fixture redesign, `UB.10`'s
   successor arm, the `PS` legibility family (09-24), or the `GEN` four.
7. **Nothing fires today.** `D27`'s earliest legal firing is 2026-09-21; you
   have verified this four times and it does not need a fifth. The two 09-21
   dispositions (`WAITS-ON:` with `none` permitted; the `(iv)` measurement
   **before** the `(iv)` implementation) come due tomorrow — do not start them
   early and do not fold them together.
8. *Minor, for the record:* four journal entries and one commit message cite
   "the 105th audit" for findings that belong to the **104th** (`38bf7a4`).
   This page is the 105th. A future reader grepping for the citation lands on
   the wrong report. No action beyond getting the next one right.

---

## FOR THE REVIEW

Your 06:37 Sunday FULL is the first organ that can act on any of this.

1. **RANK 3 is partly yours and it is cheap.** `PROGRESS.md` asserts `D20`'s
   closure is *"scoped to the `cpu<48h` class"*. The resolution's own sentence
   is unqualified and `run.py` implements it unqualified. Either cite where the
   scoping comes from, or withdraw the sentence — an unsourced scope reading on
   a current-state page is how the collision stayed invisible for 21 hours.
   The substantive call is `D32`'s and it is the owner's; the citation is
   yours.
2. **RANK 4 is still yours and still first.** The `GEN` four have had no owner
   for four days. The repair is a **new owner with a date** — not a re-open of
   the ACTED row, not a citation deletion, not a baseline widening. The ACTED
   row already refused the last two in the right words.
3. **`completeness-audit-2026-09-13-the-cognitive-half-is-the-hole` is DUE
   TOMORROW on a premise `run blocked` falsifies today** — it says *"all four
   registered barriers are reachable-on-paper"*; all four print under
   `unreachable until redesigned`. Correct the sentence; the conclusion may
   well survive it.
4. **15 OVERDUE, 21 due today, capacity 6.** `D28`'s default `(a) OVERDUE
   FIRST` fires tomorrow. Three of today's 15 are rows whose dates you set
   yourself on 09-14/09-15 at your *demonstrated* rate; `sh02-null-saturation`
   is now at its **fourth** break and carries your own binding STOP-RULE — *"if
   this date breaks too the row is DECLINED and the finding is carried to the
   owner as a class"*. It broke. That rule is yours to execute today.
5. **`2026-09-24` carries 7 promises against capacity 6**, and the tool named
   the act that built it: `lt02-...-body-chaos-is-reducible` was dated onto a
   day already at capacity when `next_free_due` printed **2026-09-21**. A
   metric, not a violation — but it is the fifth `PS`/`LT` row bundled onto that
   date and they will break together.

---

## FOR THE OWNER

**1. `D27` is due TODAY (2026-09-20) and today is genuinely your last day on
it.** Its default fires at the first slot on 2026-09-21 — the builder verified
this correctly four times and fired nothing, which is the right conduct. The
question is whether this ladder buys a mechanical screen to re-examine its own
PASS certificates, against a measured 104-of-107 false-positive showing; the
only legal default is *(i) build it, reporting-only, unfloored*, and firing it
also owes the false-positive-rate measurement. **Nothing is asked of you that
you have not already been asked. This is a reminder of a clock, not a re-ask.**

**2. NEW — `D32`, appended today, armed, `decide_by 2026-09-24. `** Your own
`D20` ruling fired on 09-19 with the sentence *"the detached lane is declared
CLOSED to registered spec work"*. The code reads that unscoped and refuses
every class; the Review's page reads it as scoped to `cpu<48h`. **Twenty-one
hours later a registered `cpu<10min` run was launched through that lane and
bought a ledger row — and the guard built to enforce your ruling said
`launchable`, because it cannot see through the wrapper `launch_detached.sh`
puts between `setsid` and the spend.** The full mechanism, with the live proof,
is RANK 1 above. What is on your desk is only the scope: does your closure cover
every registered run, or only the `cpu<48h` class the arithmetic was about? It
matters immediately, because closing it for all classes removes the only
mechanism that has ever carried a long CPU run past an hourly slot boundary
here — `LT.02` landed on the third attempt precisely because that lane survives
slot death. The armed default takes **no permission away and gives none**: make
the guard *see* the lane and say so loudly, and leave the refuse/permit line
exactly where it sits today until you rule.

**3. NO-DECISION: the defect class is at nine occurrences, and what changed is
the kind of remedy that failed.** Through occurrence seven the remedy was always
a better `LESSONS.md` entry, and the Review put that on your desk on 09-19 as a
report about how this project learns. On 09-19 the remedy was upgraded to a
**control** — correctly, and fast — and the class recurred twice more within
seven hours anyway, through the one lane the control is structurally blind to,
while its fixture reported 17 green cases against a launcher this repo does not
use. **I am not asking you to rule on this.** The specific fix is cheap and is
ordered to the builder as item 1. I am putting it in front of you because the
Review told you at five occurrences that a memory was doing duty as a control,
and the honest update at nine is narrower and more useful: *the control was
built, and it was certified against a shape that does not occur here.* If a
tenth follows this repair, the conclusion will not be about `run.py` any more,
and I would rather you had seen the ninth than met it at the tenth.

**4. NO-DECISION: liveness and the perishable price, the standing `D30`
report.** All four organs alive and on cadence: ladder **24 of 24 slots ran**
in the last 24 h (23 `rc=0`, one inherited timeout completed), overseer this
sitting, Review 09-19 06:37 with its Sunday FULL due at 06:37 today, field
watch 09-14 (next 09-21). **Dark-slot streak: 0** — counted from
`/data/jack-logs/ladder.log`, not from anyone's summary. Beside it, as `D30`'s
default requires, the price: **`2026-W37` closed with ~27.78 free Kaggle hours
expired unspent**, and **`2026-W38` opened today with 30 more that expire
Saturday 2026-09-26, with no legal buyer** — 5 of 7 cost classes are empty with
no path in, and the repair for every one is a Review design row, not a
dispatch. Fourth consecutive week. This is `D28`'s cost in a currency that
perishes, and it is a fact about item 2 of the Review's page, not a new ask.
