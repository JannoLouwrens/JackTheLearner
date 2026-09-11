# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Running history lives in git.

**2026-09-11 06:37–07:0x UTC — 88th audit.** Window: the last 24 hours
(2026-09-10 06:59 → 2026-09-11 06:5x).

## VERDICT: DRIFTING

**The ledger is sound, nothing was loosened, every ratchet is at its floor, and
the queue has zero violations.** The finding this morning is in my own output:
**yesterday this desk told the owner the builder could not run again before
2026-09-14 and handed them a command to suspend pacing. Twenty-four hours of the
same meter refute it. The gap I said was closing at 0.29 points/day closed at 5,
and `pace_gate` is on course to release the builder tomorrow, 2026-09-12 —
before the week resets and before Sunday's GPU quota expires.**

Not `INTEGRITY RISK`: no ledger row, threshold, control or certificate is
affected, and I checked all 108 PASS rows directly (section 1). The damage is to
the trustworthiness of *this organ's reports*, which is the only product it has.

---

## RANK 1 — I published a forecast that was wrong by a factor of twenty, in the direction that argued for spending the owner's budget

### What was published, 2026-09-10, by both document organs

| source | forecast | published as |
|---|---|---|
| Review `D26` addendum (`6e79f7d`) | **~6 days**; *"`pace_gate` never releases the builder at all"* | `DECISIONS_NEEDED.md` |
| **Overseer, 87th audit** (`1b693bd`) | **49 days**; *"not under ANY measured rate … off by 13×"* | `OVERSIGHT.md`, **with a `.usage-resumed` override command recommended to the owner** |

### What the meter actually did — hourly, from `ladder.log`, the file both forecasts were built from

```
                meter  line   gap
09-08T09:07       39    37     2
09-09T13:07       66    47    19     <- the gap PEAKS here
09-10T05:07       67    53    14
09-10T06:37       68    54    14     <- both forecasts written at this reading
09-10T17:07       72    58    14
09-11T01:07       72    61    11
09-11T06:07       72    63     9     <- now
```

**The gap went 14 → 9 in the 24 hours after publication: it closed at +5
points/day, not +0.29.** The meter has been **flat at 72% for 15.5 hours**
(unchanged since 2026-09-10T15:07) while the line climbs 0.39 points/hour.

### Where the error is, precisely

The forecast is a *difference of two rates*. The first term is code —
`allow = 25 + ceil(65·elapsed/100)` (`scripts/lib_usage.sh:88`) — rising **9.29
points/day**, exact, no variance. Both organs got it right. The second term is a
shared meter, and this project's own record of its daily rise reads:

    +21   +36   +9   +4     points/day

We each took **one** of those draws and subtracted. The difference ranges over
**−26.7 to +5.3 points/day** — it changes sign. A quantity whose sampling spread
is six times its own magnitude is not a rate, and a date divided out of it is not
a forecast. The confident phrasings we both reached for — *"never"*, *"under any
measured rate"*, *"off by 13×"* — are the tell.

**The 87th audit had the disconfirming evidence in its own text.** It quoted the
Review's observation that the meter was *"flat at 67% for thirteen consecutive
hourly readings"* — and then extrapolated the 24-hour delta that contained the
daylight climb as the steady rate.

### The bound that did hold, and is the only form that should be quoted

The Review's **flat-meter bound** — *"if nobody draws again, the line overtakes
the meter at T"* — said **09-11T21:06** yesterday and re-derives to
**09-12T08:40** today: a 12-hour slip over 24 hours, while both drain
extrapolations inverted. It uses only the variance-free term and states its
assumption. Re-derived this morning (meter 72%, elapsed 58%, line 63%):

```
  meter stays flat                                -> release 2026-09-12 08:40
  meter +3/day (09-10's measured external draw)   -> release 2026-09-12 18:40
  meter +4/day (09-10's measured TOTAL rise)      -> release 2026-09-12 23:40
  meter +7/day or more                            -> no release before the reset
  week resets                                        2026-09-14 05:23
```

**Under every draw rate measured since this blackout began except the two worst
single days, the builder is released tomorrow.**

### Consequences I am obliged to state against my own prior report

- **The 87th audit's `FOR THE OWNER` item 1 is WITHDRAWN as unsupported.** On
  today's reading the `.usage-resumed` override buys roughly **one day**, not
  six, at the cost of suspending pacing for the rest of the week. Materially
  different trade; the owner was entitled to see it before acting. They did not
  pull it (`.usage-resumed` does not exist) — that is luck, not process.
- **The "87% of the headroom" figure is withdrawn.** It was `2.0 ÷ 2.29` where
  the denominator was the same bad residual. The defensible form: the two
  document organs cost **1–2 points/day** against a line that advances
  **9.29/day** — **11–22% of the daily line advance**. Removing them advances the
  release by about **0.3 days** under today's dynamics, not by six.
- **The 87th audit's B4 is re-specified and its original form withdrawn.** B4
  asked for an instrument to print the point forecast. That would have made this
  worse: a wrong number computed by hand can be argued with; printed by a tool it
  inherits the tool's authority. See **B2** below.
- **What survives untouched**, because it was read from code and not from a
  subtraction: `pace_gate` is applied to the builder every slot with no
  exemption, to the overseer with a daily exemption, and **not at all** to the
  Review or field watch (`ladder_loop.sh:183`, `overseer.sh:79`, `review.sh:30`,
  `field_watch.sh:32`). The ordering question in `FOR THE OWNER` item 3 stands on
  its own evidence and is repeated below.

Corrected in the durable file as a `D26` addendum (`e112a34`) and as a lesson
(`b945a06`), with the 87th audit's lesson marked corrected in place. *(Declared
rather than done quietly: my brief permits appending to `LESSONS.md`; I added a
correction banner inside the lesson I wrote yesterday rather than only appending
a new one, on the 77th audit's mark-false-receipts-in-place precedent, because
leaving "49 days" and "87%" standing as measured facts on the page the builder
reads is the larger harm. The banner adds; it deletes nothing.)*

---

## RANK 2 — three armed defaults are overdue and unfired, and `D26` went red today

`decisions --check` is **EXIT 0** with the ratchet at its floor, and prints three
`OVERDUE — DEFAULT IS DUE TO FIRE`:

    D22   OVERDUE 2026-09-09   default writes nothing   unfired, 3 days
    D18   OVERDUE 2026-09-10   default is builder code  unfired, 2 days
    D26   OVERDUE 2026-09-11   default is builder code  unfired  <- new today

**`D26`'s overdue notice is appended to `docs/DECISIONS_NEEDED.md` by this audit
(`e112a34`) with the required wording, and its firing is routed as B1.** I am not
firing it and I am not extending any deadline: `D13` records that the overseer
may not edit its own script, and every armed default in this file's history is
stamped *"fired … (builder)"*.

**RANK 2 is materially smaller than the 87th audit made it.** That report framed
the queue as *"deadlocked on builder absence, a case the mechanism has no clause
for."* On today's arithmetic the absence ends tomorrow, so this is a **~1-day
queue, not a deadlock.** The structural observation stands — the armed-default
mechanism has no clause for an executing organ being switched off — but it did
not need the word *deadlock*, and I used it on the strength of the forecast
RANK 1 withdraws.

---

## RANK 3 — the desk's drain is unbounded and Sunday is triple-booked

`run review-queue` — **0 violations, EXIT 0** — but the metrics behind the clean
exit are the finding:

```
  46 routed: 24 OPEN, 2 HELD, 12 DISPOSITIONED, 8 ACTED, 0 DECLINED
  oldest live 18 d;  consumer last ran 1 d ago
  trailing 7 cycles:  arrived 12 (1.71/cyc)   disposed 6 (0.86/cyc)
  drain UNBOUNDED — 38 live rows, arrivals exceed disposals by 6, no projected end
  2026-09-13:  14 rows against a measured capacity of 6   !! AMBER
```

Sunday 2026-09-13 now carries **three** things at once: 14 queue rows against a
capacity of 6, the Sunday FULL Review's own anatomy and completeness audits, and
the expiry of W37's free GPU allocation. The Review declared the oversubscription
itself, a week ahead, which is the right behaviour — but declaring it does not
discharge it.

**The most important row on that day is the one nothing can see.** The Review's
`t215` disposition (`cbd6438`) found that `UnifiedBrain.py` is hashed into
`IMPL_DEPS` by **four PASS certificates** (`T2.03`, `T2.04`, `T2.06`, `T3.01`),
is **refuted by two independent FAILs** (`T2.15` at [8,9,5]/16 against a 12/16
bar, beaten by both registered bag-of-words nulls on seed 2; `T2.07` at
[2,2,2]/5), and **holds no champion seat at all** — so `champions.py`, whose
entire job is to prevent an unchallenged architectural incumbent, cannot see it,
because it checks seats that EXIST. That is the sharpest structural finding of
the week and it is scheduled onto the day that will not clear. Named here so it
survives Sunday's triage; no action of mine, seat creation is FULL-mode work.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, no findings.** 143 rows: **108 PASS, 22
FAIL, 13 VOID**. Every PASS checked directly this morning, not quoted:

- implementation resolves in `experiments/tests/` (`run.module_path_for`, file
  exists on disk) — **108 / 108**
- `commit` still resolves (`git cat-file -e <sha>^{commit}`) — **108 / 108**
- spec declares a `control` — **108 / 108**
- non-empty `control_metrics` — **106 / 108**

The two exceptions are `T0.01` and `T0.10`, both carrying
`control: "NONE, BY DECISION (52nd audit B5)"` with a stated reason — an import
either raises or it does not; a sabotaged upload fails on the service's side.
Declared and reasoned, not a silent gap. Unchanged from yesterday and correct.

**2. Thresholds and controls — CLEAN, no findings.** **Zero commits in the
window from any organ**, so nothing new to scan. Re-scanned the full 7 days of
`registry.py`, `registry_expansion.py` and `experiments/tests/` for numeric
movement. Two edits move in the loosening *direction* and both are justified:

- `MEASURED_DISCHARGE_CAPACITY 1 → 6` (`cd5a27b`) — cites six discharging commits
  by hash, calls 6 *"the demonstrated one-cycle MAXIMUM, not a sustained rate"*,
  and prints the before/after effect on its own amber count. The paired falsifier
  edits in `T0.31` (`due_pile.get(free,0) != 0` → `>= CAP`, and
  `!= [1,2]` → `!= [CAP, CAP+1]`) are the correct generalisation of the same
  constant, not an independent weakening. Justified by measurement.
- `kindless_arena_discharges 9 → 1` (`0d57b1d`) — a ratchet shrink paid for by
  *declaring eight arena kinds*, not by deleting a class; no commitment gained a
  pass credit, no seat flipped, violations 10 before and after.

`if blind or not forecast_ok:` in `t0_31…py:767` is an `or` added to a **control**
path — the blind arm must fail the property — which is strengthening, not
loosening. **No silent loosening anywhere in the window.**

**3. Drift from the goal.** The builder worked on **nothing**; it has been
switched off for 70.2 h. `demonstrated` last moved 2026-09-07T11:25 — **91.2
hours ago.** There were **zero commits from any organ in the last 24 hours**, the
first such day in this project's recent record; even the two document organs that
produced six commits yesterday produced none.

The converse question, which matters more and which no outage excuses. From
`coverage` (**EXIT 2**, the pre-existing ratcheted population, **0 commitments
with no declared spec**):

- **4 CLAIM-DEAD** — every claim spec parked or foreclosed: **smell, balance,
  shelter/building, thermal (kills)**. Three of GOAL.md's *"too cold kills him"*
  and *"he builds a shelter"* commitments are in there.
- **9 more with live claim specs and nothing passing**: touch/contact, tool use,
  told world, proprioception, plasticity, sleep, hunger/thirst, death & retry,
  fast/slow.
- The three claims GOAL.md says are most at risk of quiet neglect read
  **curiosity 2 of 12 passing**, **one brain / unison 1 of 27**, and
  learning-by-living gated behind `W0`/`W1`.
- **4 NEW unrunnable GOAL.md citations** (`GEN.02/03/06/09`, welded behind
  `LC.07` = PILOT-BLOCKED) — a standing red, correctly *not* baselined
  (`coverage.py:263` holds exactly `DP.02/DP.03/LC.04`), routed on
  `goal-cites-four-specs-that-resolve-to-corpses` DUE 09-15. Verified live, not a
  new hole.

**4. Is the builder alive and productive?** Alive, zero productive. **70
consecutive `PACING:` skips** since the last iteration ended
`2026-09-08T08:23 rc=0 — 108 -> 108 demonstrated`; every skip `rc=0`, because the
refusal and the heartbeat share a code path (86th audit's lesson, `0c367a2`).
Release forecast in RANK 1. No credit exhaustion, no repeated identical failure,
no paused loop, no abort on load.

**5. Compute honesty — no waste, and the opposite failure.** `gpu_budget.json`
has **no `2026-W37` key at all: 0.00 of 30 free Kaggle GPU-hours**, expiring
Sunday 2026-09-13. Predecessors: W32 21.06, W33 7.63, W34 1.62, W35 18.93,
W36 17.72. No GPU hour was spent without a ledger entry; none was spent at all.
`coverage` additionally reports `gpu<20min` among **3 cost classes newly empty
with no path in** — nothing runnable to implement, nothing gate-provisional to
pilot — so even a released builder has nothing registered to dispatch there, and
attempt 3's own precondition (twin-spread probe, then a committed successor gate,
then a dispatch) still binds before anything moves. **A 09-12 release leaves
roughly 24 hours against attempt 2's measured 17.61 GPU-h.** That is tight, not
impossible, and it is the builder's call to make on the row rather than to
squeeze — item B4.

**6. Stuck decisions — RANK 2 only.** `decisions --check` **EXIT 0**:
**0 `MEANS-ESCALATED`** (no fork a measurement could settle is sitting on the
owner's desk), **0 `UNDECLARED`** of 10 armed, **0 unrouted owner asks, 0
vanished owner asks**. Nothing to arm — the ratchet is at its floor, so the
standing "arm at least one per audit" has nothing to bite on. `PROGRESS.md`'s
`FOR THE OWNER` read in full and diffed against its prior revision: five items,
all five routed or explicitly NO-DECISION, none lost. No owner decision was acted
on without being recorded.

**7. Bakeoff hygiene — CLEAN, no findings.** No decision was resolved in the
window. `DECISIONS_RESOLVED.md` spot-checked across the armed-default block
(`D3`, `D4`, `D7`, `D8`, `D9`, `D11`, `D13`, `D14`, `D15`, `D16`, `D17`, `D21`):
every one carries its firing date, its executing organ, and an explicit statement
of what the default did and did not change. No VOID treated as a verdict; no
winner chosen inside a noise margin.

**Architecture (`champions --check`) — EXIT 0, and this is the week's one piece
of unambiguously good news.** The **8 `ARENA-MISSING` seats** named in my
standing brief are now **0** — `0/0 seats with a phantom arena`, and the whole
`W.1`–`W.8`, `PL.*`, `LG.*`, `LT.*` population resolves. Every other class sits
exactly at its ratcheted floor (2/3 unfalsifiable, 2+1/4 uncontestable,
2/2 unverified verdicts, 3/3 trigger debt, 1/1 kindless). The ratchet shrank the
way it is supposed to — by registering specs, never by deleting arena references.
The gap the tool cannot see is RANK 3's seatless `UnifiedBrain`.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?**

**No, and today we are not even closer to a longer list of green ticks or to a
longer record of why not.** `demonstrated` has been 108/245 for 91 hours and the
repository gained **zero commits** in the last 24. Yesterday I wrote that we were
accumulating commentary instead of evidence. Today we accumulated neither.

The thing worth saying plainly is about the commentary that was accumulated. Four
organs are healthy, fast, mutually corroborating and correct about the ledger —
and on 2026-09-10 two of them independently produced a confident quantitative
forecast that the next day's data refuted, one of them attaching a recommended
action on the owner's account. **The instruments held; the prose did not.** That
is the Review's own sentence from yesterday — *"this project's errors are
migrating out of its instruments and into its prose"* — and the cleanest
confirmation of it available is that the very next morning it caught me. The
builder is the organ this system gates, meters, ratchets and falsifies. The organ
that writes to the owner is gated by nothing but its own care, and this week that
was not enough. The creature has not moved since Monday, and the most useful
thing this desk did in 24 hours was find its own error before the owner acted on
it.

---

## FOR THE BUILDER

On today's arithmetic you wake around **2026-09-12**, with roughly 24–36 hours
before W37's GPU quota expires on Sunday. **The docket below is ordered for one
slot. Do B1 first and alone if that is all you get.**

**B1 — Fire three armed defaults. All three are overdue; all three are cheap.**
   - **`D22`** — routed by the 86th *and* 87th audits, still unfired at 3 days.
     Default **(i) THE RULE STANDS**; it writes nothing. Journal:
     *"the owner did not rule by 2026-09-08, so the pre-registered default
     fired"*, and record that it is reversible at any later date at no cost.
   - **`D18`** — overdue since 2026-09-10; notice appended by the 87th audit.
     Default **MEASURE AND REPORT, GATE NOTHING, RELAX NOTHING**:
     `lib_procwatch.sh` reads `/proc/PID/status:VmHWM` while walking pids it
     already resolves and **NAMES** any project python over the ceiling (name,
     never kill); `run_spec` records `peak_rss_mb` from
     `resource.getrusage(RUSAGE_CHILDREN)`. The ~1.5 GB figure in `SYSTEM.md`
     **stands verbatim** — not raised, not narrowed, not annotated. Journal:
     *"the owner did not rule by 2026-09-09, so the pre-registered default
     fired"*. Reversal: revert the two commits.
   - **`D26`** — **new today**; notice appended by this audit (`e112a34`).
     Default **(iv) MEASURE ONLY, GATE NOTHING, RELAX NOTHING**: `pace_gate`'s
     skip line additionally prints this project's own attributed spend beside the
     shared total, summed from `usage_ledger.jsonl`'s existing start/end pairs,
     and consecutive dark slots become a ratcheted metric. **Gate nothing, change
     no behaviour.** No spec declares `lib_usage.sh` in `IMPL_DEPS`, so no
     certificate is staled. Journal: *"the owner did not rule by 2026-09-10, so
     the pre-registered default fired"*.

**B2 — Replaces the 87th audit's B4, which was wrong. Print the BOUND and the
SPREAD, and print NO single release date.** B4 asked you to make an instrument
emit the closing rate and a projected release date. RANK 1 shows that number is a
difference of two rates whose spread is six times its own magnitude, and that
printing it would have laundered a bad extrapolation through a tool's authority.
What to add to the existing pacing print instead, as **measurement only, gating
nothing**:

   1. the line slope (exact, from the constants: `9.29 pts/day` at `PACE_FLOOR=25`,
      `PACE_CAP=90`);
   2. the current gap (`pct − allow`);
   3. the **flat-meter bound** — *"if the meter does not move, this gate opens at
      T"* — with the assumption printed in the same sentence;
   4. the meter's rise per day over the **last N days individually** (today:
      `+21 +36 +9 +4`), not averaged;
   5. **no projected release date.**

If you want one line for a human: *"gap 9; opens 09-12T08:40 if the meter holds;
the meter's last four daily rises were +21 +36 +9 +4."* That is the honest
sentence and it is one a reader can argue with.

**B3 — Count the dark slot, and name the model.** Unchanged and now asked by
three consecutive audits and the Review. On waking, append one line to
`docs/LOOP_JOURNAL.md`: consecutive slots skipped (**70 and counting**), the
`week:all models` reading that released you, and **which model actually ran**.
The third field is load-bearing: `week:Fable` is at **100%** against
`MODEL_FLOOR=95`, so you will wake on **Opus**, billed to the shared meter, and
nothing currently records that substitution where a later audit can find it.

**B4 — The GPU window is real, it is short, and the precondition still binds.**
W37 stands at **0.00 of 30** hours with the quota expiring Sunday 09-13. The
Review corrected the "W37 (opens 09-13)" date error in your priority block
(`fd2101d`) — **it closes 09-13.** Against attempt 2's measured 17.61 GPU-h a
09-12 release leaves roughly 24 hours, which is tight. **The precondition is
unchanged and still binds, in order: twin-spread result on the row → successor
gate committed in a non-dispatch commit → dispatch.** If 24 hours does not fit,
**say so on the row** rather than squeezing it — but decide that against the real
deadline, not against a phantom one and not against my withdrawn forecast.

**B5 — `W1.04` gained conjunct (c) before you register it** (Review 09-10,
`1a0e413`): 5th-percentile *measured* survival ≥ declared horizon, per-life
termination cause on the ledger row, an explicit ban on repairing it by
shortening the horizon, mechanism-disabled twin as control. Register from the
amended design, not the 09-06 text. It stales nothing — `W1.04` is not
registered.

---

## FOR THE OWNER

**1. I gave you a wrong number yesterday, and I am withdrawing it before you act
on it.** The 87th audit told you the builder would be dark until 2026-09-14 —
*"5.9 days, the largest such loss in this project's history"* — and gave you a
one-line `.usage-resumed` command to suspend pacing. **Twenty-four hours of the
same meter refute it.** The gap I said was closing at 0.29 points/day closed at
5, and on every draw rate measured in this blackout except its two worst days,
`pace_gate` releases the builder **tomorrow, 2026-09-12**, before the week reset
and before Sunday's GPU expiry.

The override still exists and is still yours. What changed is its price:
**it now buys about one day, not six.** My recommendation is that you **do not
pull it** — one day is not worth suspending pacing for the rest of the week, and
the builder's own GPU precondition needs a slot or two anyway. Full arithmetic,
the error's mechanism, and the range of measured draw rates are in
`DECISIONS_NEEDED.md` under the `D26` correction (`e112a34`).

**2. `D26` is OVERDUE and its default is due to fire.** `decide_by` was
2026-09-10 and passed unanswered; the notice is appended with the required
wording and the firing is routed to the builder as B1. **I am not extending the
deadline.** My recommendation on the substance is **unchanged** — **(i) ATTRIBUTE
THE LINE**, because the defect `D26` names (a rationing gate reading a shared
meter with no attribution) recurs every time an external consumer draws, and is
entirely independent of how long *this* blackout lasts. Priced honestly against
itself, as before: (i) resumes the builder on **Opus**, because `week:Fable` is
at 100%, billed to the meter (i) just stopped gating. The armed default remains
**(iv) MEASURE ONLY**, the only *legal* default — a default may not widen a
gate — and it fixes nothing.

**3. The ordering question stands, and it stands on code rather than on my
withdrawn arithmetic.** `pace_gate` is applied to the builder every slot with no
exemption, to the overseer with one exemption a day, and **not at all** to the
Review or the field watch. Whatever is decided about attribution, this survives
it: *when this project is rationed, which organ eats last?* Today it is the only
one that can move the creature. The quantitative case I made yesterday for how
much that costs was inflated and is withdrawn — the honest figure is that the two
document organs consume **11–22% of the daily line advance** and removing them
would advance the release by about **0.3 days**. The ordering is still nobody's
deliberate choice, and I have not proposed a number: that would be moving a gate.

**4. NO-DECISION — Sunday 2026-09-13 is triple-booked and will not clear.** 14
queue rows against a measured capacity of 6, plus the Sunday FULL Review's
anatomy and completeness audits, plus the expiry of W37's 30 free GPU-hours
(0.00 spent). The desk's drain is **UNBOUNDED**: 12 rows arrived and 6 were
disposed over the trailing 7 cycles, 38 live. Announced, not asked about — but
you should know the day is oversubscribed before it arrives rather than after.

**5. NO-DECISION — the finding I would most like you to see survive Sunday's
triage.** `UnifiedBrain.py` is hashed into the `IMPL_DEPS` of **four PASS
certificates**, is **refuted by two independent FAILs**, and **holds no champion
seat**, so the instrument built to stop exactly this — an unchallenged
architectural incumbent — is blind to it, because it audits seats that exist.
Found by the Review reading a row nobody was forcing it to read. Routed to
Sunday 09-13, the day above. Nothing here to rule on; it is builder and Review
work. I am naming it because it is the single most load-bearing structural gap
currently on the board and it is queued behind a traffic jam.

**NO-DECISION:** *The audit, section by section* is a status report. Sections 1,
2, 6 and 7 are clean and there is nothing there to rule on.
