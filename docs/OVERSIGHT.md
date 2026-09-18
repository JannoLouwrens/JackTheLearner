# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **99th audit — 2026-09-18, 12:37–13:0x UTC.** Opened at HEAD `890bf6e`.
> **The builder is back.** The 41-slot blackout ended with the weekly meter
> reset between 09:07 and 12:07 today; the 12:07 iteration ran on Fable and
> ended `rc=0` at 12:18 with six commits. It is the first productive builder
> slot since **2026-09-14T11:14** — **97 hours**.
> `demonstrated` **108 → 108**, flat since 2026-09-14T04:18 (**4 d 8 h**).
> The five overseer slots between this one and the 98th (09-17 06:37/12:37/
> 18:37, 09-18 00:37/06:37) were all `STOPPED at 91–100%`. **So were both of
> the Review's — see RANK 2.**

## VERDICT: INTEGRITY RISK

**Carried from the 98th audit, same subject, and the half-repair made one thing
worse in a way worth naming precisely.**

Scope it first, because the phrase is expensive: **the capability ledger is
clean and I re-checked it independently this morning.** 108 PASS rows; all 108
resolve to a live spec in `BY_ID`; every `commit` field resolves in git; every
one declares a control and carries `control_metrics`, except `T0.01` and
`T0.10`, whose specs declare `control = "NONE, BY DECISION (52nd audit B5)"`
with a written argument. Across seven days of `registry.py`,
`registry_expansion.py` and `experiments/tests/`, **exactly one numeric constant
moved and it moved in the tightening direction** (§2). Nothing on the
scoreboard is a lie.

The integrity risk is in the **decision register**, and it is the same two
blocks it was 44 hours ago. `docs/DECISIONS_NEEDED.md` still contains **two
different open owner decisions both numbered `D30`**, at lines 6897 and 7030.
The 98th audit's `FOR THE BUILDER` item 1 asked for two things: **renumber the
colab entry to `D31`**, and **make `parse()` refuse a duplicate id**. Commit
`a5949ae` (2026-09-17 08:30, Opus 5, owner-side session) shipped the **guard**
and not the **renumber**.

So the guard now correctly reports a HARD violation over a file it also
continues to mis-resolve — and in shipping, it **silently swapped which of the
two decisions is invisible**:

| read on | tool resolves `D30` to | invisible entry |
|---|---|---|
| 2026-09-16 (98th audit) | `due 2026-09-25` — the **colab lane** entry | the blackout escalation |
| 2026-09-18 (today) | `due 2026-09-18` — the **blackout** entry | the colab-lane ceiling |

Nobody decided that. No commit message mentions it. A reader diffing the two
audits sees an owner deadline move seven days, in the opposite direction from
the drift the 98th audit reported, with no author — which is the exact sentence
`SYSTEM.md` reserves for the deadlock armed defaults were built to replace.

---

## THE FOUR INSTRUMENTS

| instrument | exit | reading |
|---|---|---|
| `coverage` | **2** | **4 commitments with NO declared spec** — `heavy`, `far`, `tiring`, `worth-it` — and **8 CLAIM-DEAD of 29**. First time this tool has exited on the uncovered axis. See RANK 3; the number is honest and newly-visible, not a regression. |
| `decisions --check` | **1** | **1 hard violation: `DUPLICATE-ID D30`.** 0 `MEANS-ESCALATED`, 0 `UNDECLARED`, 0 `OVERDUE — DEFAULT IS DUE TO FIRE`. 1 `DEFAULT-ACTION-SAME-DAY` (D30). 4 soft `CONDUCT-MISFILED?` (D20, D27, D28, D29). |
| `champions --check` | 0 | 0 phantom arenas, 2/3 unfalsifiable, 2+1/4 uncontestable, 4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger debt, 1/1 kindless. **Every class AT its declared count; none moved since the 98th.** |
| `run review-queue` | **2** | **11 OVERDUE**, from **0** at the Review's last sitting. `review_queue_violations` 0 → 11, `review_queue_net_arrivals` 6 → 9, both recorded at HEAD. See RANK 2. |

**`D19` no longer prints OVERDUE** — the owner ruled it on 09-17 (`ab17202`),
and the builder's journal correctly records that it did **not** fire the armed
default because a ruling overtakes one. That is the right call and I checked it
rather than assuming it.

**No `MEANS-ESCALATED`, no `ARENA-MISSING`, no `NO-ARENA` regression, no
`UNDECLARED` to arm.** I am required to arm at least one open decision per
audit and there is none to arm: every open entry carries a class, and the four
`goal`-class entries carry a `default` and a `decide_by`. I am reporting that
rather than manufacturing an entry to have something to do.

---

## RANK 1 — The `D30` duplicate is unrepaired 44 hours on, and the guard that shipped without the renumber moved the invisible entry instead of removing it

**The file, unchanged:**

```
$ grep -n '^DECIDE: D30' docs/DECISIONS_NEEDED.md
6897:DECIDE: D30      # the blackout escalation   — decide_by 2026-09-18  (TODAY)
7030:DECIDE: D30      # the colab-lane ceiling    — decide_by 2026-09-25

$ $PY -m experiments.decisions --check | grep -E '^\s+D30'
    D30    costs   0 specs   due 2026-09-18   [SAME-DAY RACE: must fire before its own 2026-09-19 event]
```

**What is now invisible, and it is not nothing.** The colab-lane entry is the
97th audit's own finding: `remaining()` returns infinity for the colab lane, it
has no ceiling, no overrun mark and no refusal. Its armed default is
**`(i) MARK BUT DO NOT CAP`**, `decide_by` **2026-09-25**. Because `parse()`
resolves `D30` to the blackout block, that default **cannot be seen, cannot go
`OVERDUE`, and cannot fire**. Its `safety_hazards` and `firing_diff` checks now
run on the blackout text and have never read it.

**The owner-facing consequence is already realised, and it is on the page the
owner reads.** `docs/PROGRESS.md` `FOR THE OWNER` item 1, written by the Review
on 09-16, reads:

> **1. `D30` — cited, not re-asked (`decide_by` 2026-09-25).** The builder is
> now dark **41 consecutive slots** …

That paragraph describes the **blackout** decision and quotes the **colab**
decision's date. The Review did not invent it — it read the tool, and on 09-16
the tool said 09-25. The blackout entry's real `decide_by` is **2026-09-18 —
today**. The owner has been handed a deadline a week late on a decision that
expires today, and **`PROGRESS.md` is stamped STALE, so it will not self-correct
until the Review next sits.**

**The repair is unchanged from the 98th audit and it is one line of Markdown:**
renumber the **colab** block (line 7030) to **`D31`**, keep `D30` = the blackout
entry because that is the id `PROGRESS.md` gave the owner, and do not delete
either. The guard is already in place and will confirm the fix by going quiet.

### RANK 1b — a second-order effect of the guard, reported at its true scope

When `check_rc` is nonzero and no ratchet is broken, `decisions.py:2199-2202`
takes the `elif rc:` branch and the `ratchet ok (…)` line is **not printed at
all**. That line is the only place `UNROUTED-OWNER-ASK` and
`VANISHED-OWNER-ASK` are displayed — the two readings the 69th audit's B2 built
precisely because the Review's owner-asks roll off `PROGRESS.md` unanswered in
24 hours.

**Stated honestly and no wider: this is a display loss, not a gate failure.**
Both counters remain in `ratchet_debt()`, so *growth* still trips
`RATCHET BROKEN` and still stops the gate. What is lost is the healthy reading.
It matters today only because today is the one day it could bite — the register
has a hard violation *and* `PROGRESS.md` is frozen STALE, so an ask could vanish
from a page nobody rewrote while the counter that watches for it prints nothing.
I read the page's `FOR THE BUILDER` and `FOR THE OWNER` sections by eye instead;
both are intact, 5 owner items, none unrouted, none vanished.

---

## RANK 2 — The Review has been gated out for two consecutive days, its queue went 0 → 11 OVERDUE, and the cause is a schedule shape, not a backlog

```
2026-09-17T06:37:03  review: STOPPED at 91% weekly usage
2026-09-18T06:37:03  review: STOPPED at 100% weekly usage
2026-09-18T~10:00    weekly meter resets
2026-09-18T12:07:10  builder: iteration start — 22% of meter, runs normally
```

**The builder is polled hourly and got its slot back the same day the meter
reset. The Review is polled once a day, at 06:37, and the reset landed after
it.** So the desk that disposes the queue is dark until **2026-09-19 06:37** at
the earliest, on a box that has had abundant budget since roughly 10:00 this
morning. `scripts/review.sh:30` calls `usage_gate say || exit 0` and there is no
retry, no deferred slot and no catch-up path anywhere in the script.

**The cost is already on the instrument.** All 11 OVERDUE rows are Review-desk
design debt — none names a builder unit — and the Review itself wrote on 09-16
that three of them were *"my bill"* and would be paid on the date they carried.
It could not pay them: it was switched off at 06:37 on both mornings.

```
OVERDUE (promised → days late)
  2026-09-16  t211-diayn-metric-cannot-separate-mi-from-noise          2 d
  2026-09-16  five-commitments-are-claim-dead-behind-foreclosures      2 d
  2026-09-16  hr5-fixture-refuted                                      2 d
  2026-09-16  t108-noise-floor-is-quoted-by-nobody                     2 d
  2026-09-16  t108-bar-set-from-n1-is-now-the-projects-largest-blocker 2 d
  2026-09-17  t309-control-clears-the-claims-own-margin                1 d
  2026-09-17  waits-on-declared-field                                  1 d
  2026-09-17  so10-tie-break-hands-the-seat-to-an-ineligible-arm       1 d
  2026-09-17  hash-salt-lottery-in-a-gated-metric                      1 d
  2026-09-17  lg13-champion-makes-lg10s-invariance-conjuncts-structural 1 d
  2026-09-17  oversight-for-the-builder-has-no-reader                  1 d
```

**The DUE-DATE PILE makes tomorrow worse, not better.** The consumer's best
measured cycle discharged **6** dated rows. Tomorrow's sitting inherits 11
OVERDUE plus 3 more falling due on 09-19 — **14 against a measured capacity of
6** — and the tool's own advice, *"next date with room … 2026-09-19"*, is now
wrong because it was computed before the desk missed two days.

**This is `D28`'s subject arriving from the other side.** `D28` (`decide_by`
2026-09-21) asks whether the desk is keeping up, and its armed default `(a)
OVERDUE FIRST` orders the desk's *own* sitting. Neither the entry nor its
default contemplates the desk **not sitting at all**. A desk that is ordered to
pay its overdue class first, and is then denied the slot in which to do it, has
a deadline that moves for a reason no organ records.

**This is builder work, not owner work, and I am routing it that way.** It is
organ mechanics — `conduct`, in `SYSTEM.md`'s three classes — and it is
repairable without touching a threshold: see `FOR THE BUILDER` 2.

---

## RANK 3 — Four of `GOAL.md`'s own constitutional commitments have ZERO specs, and `coverage` exits 2 on that axis for the first time

```
  heavy     0 specs  0 pass  0 now  1 nominated (PG.2)   NO SPECS
  far       0 specs  0 pass  0 now  0 nominated          NO SPECS
  tiring    0 specs  0 pass  0 now  0 nominated          NO SPECS
  worth-it  0 specs  0 pass  0 now  0 nominated          NO SPECS

  4 commitment(s) with NO declared spec, 8 CLAIM-DEAD … EXIT 2
```

**This is the highest-priority *ladder* finding by my standing rule, and it is
simultaneously the best thing that happened today.** The Review found the hole
on 09-16; the builder closed the instrument half this morning (`2fd7de5`), and
I checked the commit rather than trusting the message:

- The diff to `experiments/coverage.py` is **purely additive** — four new
  `COMMITMENTS` entries, +12 lines, nothing removed, no regex widened.
- `claim_dead` **4 → 8**, recorded through `run ratchets record` in the same
  commit, and the commit message declares the growth **in advance** as the gap
  becoming visible.
- `T0.21` re-bought PASS (`16927ed`) with `properties_failed 0.0` and the
  control still failing 7 of 12 named properties. Its spec was **not** touched —
  `registry.py` has no change in the last 24 h. `commitments_uncovered` went
  `0.0 → 4.0` inside the **experiment** metrics and the spec passed anyway,
  because `T0.21` asserts the register is *well-formed*, not that it is *full*.
  **Nothing was loosened to let this land.**

That is the ratchet working exactly as designed: the number got worse because
the world got more honest.

**What is still owed is the ladder half, and it is not the builder's to choose.**
Four commitments with no spec are invisible to `run blocked`, to `champions`, to
every ranking this project owns. `8 of 29` commitments are now CLAIM-DEAD —
`smell`, `balance`, `shelter/building` and `thermal (kills)` each sit behind a
PARKED spec whose declared successor is itself `PILOT-BLOCKED` or
`VOID-FORECLOSED` (`park_release_pairs = 3`, unchanged 15 days), plus these
four. **The fork — write falsifiable claims for the four, or correct
`GOAL.md`'s sentence to name only what we intend to test — is the owner's, and
the Review has it on a dated row it could not sit to answer** (RANK 2). I am
not pre-empting it; I am recording that the row is now 2 days overdue and its
answer is what unblocks a quarter of the constitutional register.

---

## RANK 4 — 26.5 perishable GPU-hours expire in ~36 h, and the two counters built to police GPU waste still cannot see the class of spend that wasted it

**Budget.** Week `2026-W37`: **3.4899 h charged of 30 — 26.51 free, expiring
Saturday 2026-09-19.** Lane split: colab 2.1109, kaggle 1.3790. The running
probe (pid 1529308, authorised ~1.05 h, 24 min elapsed at my read) will add to
colab at harvest.

**The authorised buyer is still unbought.** `T1.07`'s re-buy — **~0.47 GPU-h**,
certificate deliberately staled since 2026-09-14T11:13 — is 4 days old and was
deferred again this morning, correctly, because the probe holds the GPU
serialisation lock. It is named as the first unit of the next slot. At an hourly
cadence with ~36 h left that is affordable; I note it because it is the only
cheap unit on the board that moves a certificate, and this week's quota has
already lost 41 slots to a meter three-quarters of which is not this project's.

**Two honesty gaps, both flagged by the 98th audit, both untouched:**

1. **`overruns` is `[]`.** Both 09-14 colab jobs declared `est_hours: 0.7` and
   billed **1.03** and **1.08** — **47%** and **54%** over — and neither left a
   mark, because `charge()`'s overrun path is kaggle-only (`gpu.py:~516`).
2. **`gpu_hours_no_verdict` reads `T1.08: 0.36 h / 1 attempt / 1 verdict`** —
   it does not contain the **2.11 colab hours that retrieved nothing**, because
   the counter joins charged jobs against `ledger.results` and a probe is
   *defined* as buying no ledger row. The one instrument built to catch
   GPU-with-nothing-to-show is structurally blind to the one class of spend
   guaranteed to have nothing to show.

`gpu_hours_no_verdict` TOTAL is **48.42 h**, of which **`D1.0` 33.78 h across 2
attempts and 0 verdicts** — unchanged since 09-13 and still the largest single
block of unredeemed compute this project has. `gpu_unattributed_jobs = 21`, AT
floor.

---

## RANK 5 — Three carried builder items are untouched because `ladder_loop.sh` has not changed in seven days

`git log --since='7 days ago' -- scripts/ladder_loop.sh` is **empty**. That
leaves the 97th and 98th audits' structural asks all still open:

- **The pace-skip path still cannot notice a finished detached artifact.**
  `HARVEST_PATHS` is four in-repo files; the probe writes to `/data/` by design.
  This is what left `/data/t108_backend_probe.json` sitting unread for 43 h.
- **`declared_pids` is still not pruned on exit.** Right now it declares
  `1530368  run_spec T0.21` — a process that exited at 12:18. A declaration that
  outlives its process is a claim that outlived its evidence, and the next
  waking slot reads it as work in flight. (`1529308` is genuinely live and
  correctly declared; I verified it with `ps`.)
- **`decisions --check`'s OVERDUE read is still not on the skip branch**, under
  the comment already written at `ladder_loop.sh:112`. This one is less urgent
  than it was — the owner ruled `D19` by hand on 09-17 — but the mechanism that
  made `D19` sit overdue for 31 skipped slots is unchanged.

These are not new findings. They are the same findings, and the honest reason
they are unrepaired is that the builder has had **one** slot in 97 hours and
spent it on the perishable head, which was the correct ordering.

---

## RANK 6 — Four open owner decisions are organ-mechanics questions. Soft, reported, not charged.

`decisions --check` flags `D20`, `D27`, `D28`, `D29` as `CONDUCT-MISFILED?` —
class `goal`, blocking no spec id. All four are about how the **organs** work
(a detached-lane wall, a PASS re-examination screen, the Review's throughput, a
seat's missing diagnostic), not about what Jack must **become**.

**I am not calling this a violation, and here is why.** `c7052fa` reclassed six
of seven such entries on 09-17 — the desks did exactly what the flag asks. Each
of these four remaining entries contains an explicit, written argument for why
it is escalated rather than executed: `D29`'s is *"if you believe an
architecture seat must carry its declared guards to hold its marking at all,
then (iv) is your answer and it is the reason this is on your desk"*. That is a
`goal`-shaped question wearing `conduct` clothes, and the tool says the flag is
*"a question about routing, never a blocker."*

What I will say: all four are **armed**, with a `default` and a `decide_by`, and
none is `MEANS-ESCALATED`. The `D1` disease is not present. This is the
low-temperature version of it and it is being managed.

---

## THE AUDIT, SECTION BY SECTION

**1. Integrity of the ledger — NO FINDINGS.** 108 PASS. All 108 resolve in
`BY_ID`. Every `commit` field resolves to a live object in git (0 missing). Every
PASS declares a `control`; 106 carry `control_metrics` and the 2 that do not
(`T0.01`, `T0.10`) declare `control = "NONE, BY DECISION (52nd audit B5)"` with
the argument written into the spec. No PASS is a claim without evidence.

**2. Thresholds and controls over seven days — NO FINDINGS, and the one move
was a tightening.** `MIN_DISTRACTOR_EVAL` **30 → 59** in `ME.1`/`ME.3`/`ME.5`
(`db4200ec`), with the arithmetic in the commit message: a perfect run over `m`
negatives certifies only `a_L = γ^(1/m)`, so `m ≥ 59` is the γ=0.05 minimum that
can certify the 0.95 bar at all; the prior 39.3 ± 2.9 certified only 0.927. **The
0.95 bar itself is untouched in both directions.** The only other edit to a test
in 24 h is `t1_08_seed_variance.py`'s `print("DONE", json.dumps(out)[:600])` →
`print("JACKRESULT", json.dumps(out))`, which removes a truncation that
destroyed data and touches no constant. No `_check` gained an `or`. No seed count
fell. No assertion was removed. No control was deleted or weakened.

**3. Drift from the goal — none in the builder's slot; the converse is RANK 3.**
Everything the builder did today traces:
- `425a7e3` T1.08 stdout repair → *"Really learning, not appearing to learn"* —
  it is the machinery that decides whether a single-seed number is quotable.
- `2fd7de5` + `16927ed` coverage register + T0.21 → `GOAL.md:186-188` directly.
- `bb823e0`, `890bf6e` receipts and journal → *"protects the honesty of watching
  what happens."*

Nothing served none. The harder converse: **`one brain / unison` has 1 PASS
across 27 specs; `curiosity` 2 across 12; `fast/slow` 0 across 8, five of them
welded behind `LC.03`'s VOID; `sleep` 0 across 5; `plasticity` 0 across 4.**
The thesis families are still the thinnest part of the ladder, and that has not
changed in a week.

**4. Builder alive and productive — ALIVE, and today it produced.** 1 iteration
in 24 h (12:07 → 12:18, `rc=0`), preceded by 2 `ABORT: usage unreadable` and 21
`STOPPED at 100%`. 6 commits, all pushed. **PASS delta 0** — the one settle
event was a `T0.21` **re-buy**, not a first-ever verdict or a status change.
Last capability-moving verdict is still `T1.08 FAIL`, **2026-09-13, 5 days ago**.
No repeated identical failure, no paused loop nobody resumed, no credit
exhaustion now that the meter reset. `run next` reads **1 fresh of 44** — `HR.1`,
*"The voice corpus is honest before anyone is scored"*, `cpu<10min`, needs
implementing. There is a cheap, legal, ladder-moving unit on the board for the
first time in days.

**5. Compute honesty — RANK 4.** 26.51 free of 30, expiring in ~36 h; 2.11 colab
hours retrieved nothing and left no overrun mark and no entry in the counter
built to find exactly that.

**6. Stuck decisions — RANK 1 and RANK 6.** Nothing on the owner's desk has
enough evidence to be decided that the system could have settled by bakeoff; no
`MEANS-ESCALATED`. One owner decision was acted on and **was** recorded
(`ab17202`, D19). One (`D30`-colab) is stuck because nothing can see it.

**7. Bakeoff hygiene — NO NEW FINDINGS.** `DECISIONS_RESOLVED.md` shows 12
decisions resolved by armed default since 09-01, each naming its option and its
firing slot. No decision was made without a gate in the window. The standing
concern is unchanged and already ratcheted: the **Learning core** seat is held
**BY VERDICT** on `LC.03 = VOID`, and `champions` reports it as
`VERDICT-IS-A-VOID` with all three re-open triggers unreachable. A VOID is being
carried as a verdict — it is counted, it is disclosed, it has a queue row and a
decision (`D29`), and it has not moved in 17 days.

**8. The honest summary.** *Are we closer to a curious humanoid that climbs the
ladder?* **No — but for the first time in four days the answer is "not yet"
rather than "we could not have been."** The builder came back after 97 hours and
spent its one slot well: it repaired the mechanism that was destroying a
measurement, and it made a five-week-old blind spot in the goal register
countable. Both of those are the work of a system that is trying to be honest
with itself. Neither is Jack. `demonstrated` is 108 for the fifth day.

The thing I would fix first is not on the ladder at all. Two desks and a builder
are now gated by a shared meter three-quarters of which belongs to other tenants,
and the gate's shape — hourly for the builder, once-a-day-or-nothing for the
Review — means the organ that *disposes* work loses whole days that the organ
that *creates* it does not. Eleven broken promises accumulated in exactly that
gap. We have built a very good machine for noticing that we are stopped. The
next thing worth building is one that does not lose a day to a meter that has
already reset.

---

## FOR THE BUILDER

Ordered by what stops being repairable first.

1. **Renumber the colab `D30` block to `D31`. One line.** `DECISIONS_NEEDED.md`
   line **7030**: `DECIDE: D30` → `DECIDE: D31`, and the `## D30 — The colab GPU
   lane has no ceiling…` heading at line **6970** with it. **Keep `D30` = the
   blackout entry** (`decide_by` 2026-09-18) — that is the id `PROGRESS.md` gave
   the owner, and reassigning it under them is not yours to do. **Do not delete
   either entry.** The guard `a5949ae` added is correct and will confirm the fix
   by going quiet; it was shipped without this half and that is the whole of
   RANK 1. Verify with `grep -c '^DECIDE: D3[01]'` → 2 distinct, and
   `decisions --check` printing **two** rows where it prints one today.

2. **Give the Review a retry path. This is RANK 2 and it is the highest-value
   structural repair on the board.** `scripts/review.sh:30` is
   `usage_gate say || exit 0` on a `37 6 * * *` crontab: one poll a day, no
   retry, so a meter reset landing after 06:37 costs the desk a full day even
   with budget abundant. The builder is polled hourly and lost nothing today;
   the Review lost two days and 11 promises. **The minimal honest fix is a
   deferred slot, not a wider gate**: when `usage_gate` refuses, record the
   missed sitting, and let a later cron poll the same day run the DAILY exactly
   once if and only if the sitting has not happened yet. **Do not raise, lower
   or re-base the 90% stop, and do not exempt the Review from it** — the gate is
   the owner's rule and this changes only *when it is asked*, never *what it
   answers*. `review_liveness` already detects the missed sitting and stamps
   `PROGRESS.md` STALE; what it cannot do is give the day back.

3. **Prune `declared_pids` on exit, third asking.** `1530368  run_spec T0.21`
   declared at 12:17, process gone by 12:18, still declared at 12:37. Reap on
   exit or stamp `EXITED <ts>`. `1529308` is genuinely live and correctly
   declared — the file is not wrong, it is stale, and the next waking slot
   cannot tell those apart.

4. **Mark per-job GPU overruns on every backend.** `gpu.py:~516`'s overrun path
   is kaggle-only. Both 09-14 colab jobs declared `est_hours: 0.7` and billed
   1.03 and 1.08; `overruns` is `[]`. **This is a report, not permission to
   invent a colab ceiling** — that number is the owner's, and it is the subject
   of the decision currently numbered `D30` that nothing can see (item 1).

5. **Make PROBE hours visible to `gpu_hours_no_verdict`.** It joins charged jobs
   against `ledger.results` (`run.py:~1456`); a probe buys no ledger row by
   definition, so 2.11 colab hours that retrieved nothing read as zero waste.
   Extend the join to read `gpu_submissions.jsonl`'s `spec_phase` into a named
   `PROBE` bucket. **Reporting-only and unfloored**, per `D27`'s reasoning:
   probe spend is legitimate and gating it would punish the honest thing.

6. **The pace-skip path must notice a finished detached artifact.**
   `HARVEST_PATHS` is four in-repo files; detached runs write to `/data/`. Check
   `declared_pids` for an exited pid whose declaration names a dispatch and
   `say` it into `ladder.log` loudly. **Do not widen `git add`** — the `add -A`
   ban stands, and committing the interpretation still belongs to an unpaced
   iteration. Noticing costs nothing.

7. **`T1.07`'s re-buy, ~0.47 GPU-h, off a quota that expires Saturday.**
   Carried, and correctly deferred this morning behind the probe's GPU lock.
   First unit of the next slot, as your own journal says.

8. **Do not touch the four uncovered commitments' ladder half.** `heavy`, `far`,
   `tiring`, `worth-it` need either falsifiable claims or a `GOAL.md` correction,
   and that fork is the Review's row and the owner's call (RANK 3). Your half —
   the register — is done and was done well. `HR.1` is the fresh runnable unit
   if you want ladder movement.

---

## FOR THE OWNER

**1. `D30` expires TODAY and you have been given the wrong date for it.** The
blackout escalation's real `decide_by` is **2026-09-18**. `docs/PROGRESS.md`'s
`FOR THE OWNER` item 1 tells you **2026-09-25**; that is the *other* `D30`'s
date, and the Review wrote it in good faith from a tool that was mis-resolving
the duplicate. **Nothing new is asked and no ruling is needed from you to fix
the register** — the renumber is routed to the builder as item 1. What you
should know is that its armed default `(v) REPORT THE STREAK, GATE NOTHING,
RELAX NOTHING` fires **tomorrow, 2026-09-19**, and that the decision's own text
states its price plainly: it cannot save this week's 26.51 perishable GPU-hours,
because those expire the same day. **The blackout it was written about has
ended** — the builder ran normally today on the reset meter. The question it
asks, whether this project's pace should be measured against a meter it draws
roughly a tenth of, is unchanged by that and is still yours.

**2. NO-DECISION — the Review's missed sittings, reported because it is the
shape that hides.** The Review was gated out at 06:37 on both 09-17 and 09-18,
and the weekly meter reset a few hours *after* today's slot. Its queue went **0
→ 11 OVERDUE** in that window and tomorrow's sitting inherits **14 dated rows
against a measured capacity of 6**. **I have routed the repair to the builder as
organ mechanics (`conduct`) rather than putting it on your desk**, because it is
a question of when a script is polled and not of what this project is for, and
it can be fixed without moving the 90% stop by a single point. I am telling you
because `D28` — *"is the Review keeping up?"*, `decide_by` **2026-09-21** — is
already on your desk, and the evidence under it has changed character: the desk
is not merely behind, it was **switched off**. If you rule on `D28`, rule on it
knowing that two of the three days in its evidence window contained no sitting
at all.

**3. NO-DECISION — `GOAL.md`'s seven survival primitives, now countable for the
first time.** *"Survival earns him the primitives that make anything else mean
something — hot, heavy, far, tiring, dangerous, worth-it, that-person-lied."*
**Two are certified** (`PS.03` dangerous, `LG.02` that-person-lied), **one is
CLAIM-DEAD** (hot), and **four now read `NO SPECS`** in an instrument that could
not see them until this morning. `claim_dead` is **8 of 29** commitments. The
fork — commit to falsifiable claims for the four, or correct the sentence to
name only what we intend to test — **is yours and it is not ripe today**: a
default may not narrow what this project has promised itself, so it cannot be
armed the usual way. The Review holds the dated row and could not sit to answer
it (item 2). **Nothing was deleted to make a number look better**; the ratchet
was allowed to get worse on purpose, and I verified the commit that did it.

**4. NO-DECISION — the ledger is clean and I want that on the record beside the
verdict.** 108 PASS rows, every commit resolving, every control declared, and
across seven days the only threshold that moved was a memory-abstention
denominator that moved **upward** with its arithmetic shown. The INTEGRITY RISK
above is about the **decision register**, not the scoreboard. No capability this
project claims is in doubt this morning.
