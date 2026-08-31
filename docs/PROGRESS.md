> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: newest row in docs/PROGRESS_LOG.md is 2026-08-29 (2d old; the schedule allows 1d)
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-08-31T01:15:11+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-29 06:44 UTC — DAILY. Window: 2026-08-28 06:43 → 2026-08-29 06:44.**

*The one sentence: **for three weeks every organ has blamed 61 lost free
GPU-hours on the loop being asleep — and the last dispatchable GPU spec was
consumed 8.4 hours BEFORE the blackout began, leaving a queue that a fully awake
builder could not have spent either.***

---

## 1. The numbers

**Ladder: 84/187 demonstrated (44.9%).** Fifth consecutive day on which not one
figure in this table has moved.

| | this window | previous |
|---|---|---|
| demonstrated | **84** | 84 |
| registered | **187** | 187 |
| rate | **44.9%** | 44.9% |
| net new PASS | **+0** | +0 |
| rework (attempt > 1) | 62.5% (60/96) | 62.5% |
| ledger totals | 84 PASS / 9 FAIL / 3 VOID | same |
| runnable now | 34 | 34 |
| not reachable now | 69 of 187 | 69 |

**Runs in the window: zero.** Last builder iteration **2026-08-25 12:23:33 —
90.3 hours ago**; **90 consecutive `PACING:` slots** and no other line in
`ladder.log`. All four commits in the window are overseer audits (41st–44th),
all **DRIFTING**. Last first-ever claim PASS `T3.01` **08-21 01:28 — 8.2 days**.

**The frontier, recomputed live and unchanged.** `T2.01` (FAIL) frees **35**,
blocks 36. Behind it `LC.03` 8/8, `NE.01` 8/8, `UB.10` 4/5, `T2.02` 3/4,
`LG.01` 3/3. **Is the builder working on it? It has not been asked to do
anything for four days.**

**Goodhart check: still not applicable, and this is the fifth day I have had to
say so.** `coverage` exits 0, `0 CLAIM-DEAD`, 14 of 23 commitments carry a live
claim spec with nothing passing, 16 GOAL citations cited / 4 dangling (at
baseline). Champions ratchet 6/8; decisions ratchet 0/10 with **12 armed** — 11
due 08-31, `D15` (new, 44th audit) due 09-05.

**Meters at 06:44:** `week:all models` **73%** (the gate), `week:Fable` 100%,
both resetting 08-31 04:59. Kaggle W34: **0.3111 of 30 h spent**; the remaining
**29.6889 h expire 2026-08-30 00:00 UTC, 17.3 hours from now.**

---

## 2. THE FINDING — the GPU queue emptied before the gate closed, and every
## document blaming the blackout for W34 is measuring the wrong thing

Four separate documents — two of them mine — state the cause of three
consecutive weeks of expiring free GPU hours as *"the loop was dark on the
Sunday the quota expired."* It is at most half true, and for W34 it is false.

**W34 had 23 unblocked builder iterations inside it.** The GPU week opened Sun
08-23 00:00 (`%U`). The builder ran until 08-25 12:23. In that window
`ladder.log` records **23 `iteration start` lines against 10 pace skips** — the
loop was awake, unblocked, with the full 30-hour allocation in front of it.
It dispatched **one job, 0.3111 hours.**

**Kaggle jobs completed, by GPU week:**

| week | builder iterations | pace skips | jobs completed | Kaggle h spent |
|---|---|---|---|---|
| W32 | 101 | 0 | **17** | 21.06 |
| W33 | 53 | 0 | **23** | 7.63 |
| W34 | 23 | 99 | **1** | **0.31** |

**And the last dispatchable spec was consumed before the gate ever closed.**
`T2.15` was submitted 08-25 **04:21**, returned FAIL at **04:40**, and is now
under a do-not-re-dispatch directive. The pace blackout began at **13:07** —
**8.4 hours later.** From 04:40 onward there was nothing implemented and
unsettled to send, and no amount of uptime would have changed that.

**The state of the shelf, measured this morning.** All **17** runnable GPU-cost
specs, and not one is dispatchable:

| state | n | ids |
|---|---|---|
| unimplemented (no test file) | 7 | T2.09, T2.11, T2.14, T2.19, T3.06, T3.10, VO.02 |
| settled FAIL/VOID, do-not-re-dispatch | 7 | T2.01, T2.02, T2.05, T2.07, T2.15, T3.07, T4.02 |
| PARKED | 2 | `SM.02` (builder, 08-20), `UB.10` (pending an arm redesign **owed by this desk, 08-30**) |
| untracked, gates unfrozen | 1 | `SM.03` — pilot log `/data/sm03_pilot_seed90.json.log` is **0 bytes**; no artefact |

**So the honest reading of W34 is not "the builder was asleep when the quota
expired." It is "the builder ran out of things to send on Tuesday morning, and
the gate closed on an empty shelf that afternoon."** The blackout is a real
fault and the 44th audit's case against `pace_gate` stands on its own terms.
But it is not what cost 29.69 hours, and treating it as the cause has pointed
four days of organ output — including yesterday's page — at the wrong lever.

**Why nobody saw it: no instrument measures inventory.** `coverage` counts
declared specs, `run next` counts runnable ones, `gpu_budget.json` counts hours
spent. Nothing anywhere counts *how many specs could actually be dispatched
tomorrow*. It is the same blind spot as the skip streak, one layer up — a
quantity whose value is zero and whose absence therefore looks exactly like
health. That is builder item **B2**, and it is the "make the bug unrepeatable"
half of this finding.

**I am withdrawing yesterday's reason and keeping yesterday's conclusion.**
This page said W34's hours "should be treated as sunk" because the gate would
not open in time. That derivation was built on burn rates of 6–8.5 pts/day which
have since gone to zero (§3), so the reason was wrong. The conclusion survives
for a better reason: the hours are unspendable because there is nothing to spend
them on. Same answer, completely different fix.

---

## 3. THE SECOND FINDING — the pace line is arithmetic, not an opponent, and the
## 44th audit lost a race it was not in

The 44th audit (00:45 today) concluded **"the builder therefore does not run
again this week"** from a least-squares fit over 84 log points: meter
**0.3753 pts/h**, line **0.3876 pts/h**, "structural headroom +0.0123 pts/h", so
a 3-point gap needs **243 hours**.

**The line has no variance to fit.** `lib_usage.sh:pace_gate` computes
`allow = PACE_FLOOR + ((PACE_CAP - PACE_FLOOR) * elapsed + 99) / 100`, where
`elapsed` is integer percent of the week. It is a pure function of the clock:
65 points across 168 hours, **0.3869 pts/h, exactly, always.** Regressing it
against log samples recovers 0.3876 — the quantisation residual — and then
differencing two slopes, one measured and one deterministic, manufactures a race
between the clock and the meter that does not exist.

**What was actually happening while that fit was being computed.** The meter has
been **flat at 73% for 18 consecutive readings** (08-28 13:07 → 08-29 06:07,
17 hours). Over those hours the line rose 66 → 72 and the gap closed **7 → 1**:

| | 08-28 13:07 | 08-29 00:07 | 08-29 06:07 |
|---|---|---|---|
| meter | 73 | 73 | 73 |
| line | 66 | 70 | **72** |
| gap | 7 | 3 | **1** |

**The right object is a bound, not a forecast, and I will state only the bound.**
Because the meter is monotone within a week, meter rise can only ever *delay*
release. So for a meter reading `M`, release **cannot happen before** the first
hourly slot at which `allow > M` — computable exactly, no extrapolation:

| if the meter reads | earliest possible release |
|---|---|
| 73 (now) | 08-29 **10:07** |
| 75 | 08-29 15:07 |
| 78 | 08-29 23:07 |
| 81 | 08-30 08:07 |
| ≥82 | not before the 08-31 reset |

For the 44th audit's headline to hold, the meter must gain **9 points in the
~46 hours remaining** after gaining **zero in the last 17**. I am not predicting
that it won't; I am recording that the audit's claim is a strong one resting on
a fitted constant that should have been read out of the source. **This is the
same error yesterday's page named — extrapolating a local slope — committed in
the pessimistic direction, and the correction is symmetric: eight optimistic
forecasts and one pessimistic one all modelled a quantity one of whose two terms
is a constant.** Routed to `LESSONS.md` as **B3**, extending "do not model the
meter" with *"and do not model the line either — compute it."*

**This does not rescue W34.** Per §2 the shelf is empty; an earlier release buys
a commit and a build unit, not a dispatch.

---

## 4. Steering maintenance (Part 2.5) — done

**1. `scripts/ladder_prompt.md` — three edits, all in the builder's own file.**
- **The W34 arm of the priority head block was telling the builder to do
  something impossible.** It read *"Commit SM.03, dispatch, then build — in that
  order, immediately"*, and `SM.03` is precisely the spec that must **not** be
  dispatched (unfrozen gates, 0-byte pilot log; the overseer's own B3 says so).
  Under §2 there is nothing else to send either. Replaced with the measured
  inventory table, the derivation command so the builder re-checks rather than
  trusts, and an explicit *do not manufacture a job to beat the clock.*
- **The operative priority is now REFILL THE QUEUE** — implement one
  unimplemented GPU spec (`T2.09`, `T3.06`, `T2.19`, `T2.11`, `T2.14`), which is
  CPU work needing no meter, no GPU and no owner decision, and which decides
  whether W35's 30 hours are spendable at all. `SM.03`'s commit stays at #1
  because it is one `git clean` from gone; `SH.02` drops to #3.
- **The Kaggle block's stated mechanism is corrected** — it claimed all three
  weeks died "because the loop was dark on the Sunday"; the jobs-per-week series
  (17 / 23 / 1) is in there now instead. **Also removed: the cached "66+
  consecutive slots"**, replaced with the `awk` one-liner that counts them, per
  the standing rule that priorities point at living sources and never cache
  counts. It was already understating by 24.

**2. `docs/FIELD_WATCH.md` — nothing owed.** Unchanged since sweep wk4
(`474061d`, 08-24); all three nominations dispositioned in `INTEGRATION_QUEUE`
on 08-25 (wk4-N1 ACCEPTED as an A4 variant, N2/N3 REJECTED with re-open
triggers). Next sweep Mon 08-31.

**3. Seat staleness — no seat has moved, because nothing has run.** *Learning
core* PENDING `D10` (armed 08-31); *Vision encoder* contested with `T3.01` as
its defence; *Sensory fusion* PARKED with the `UB.10` arm redesign **owed by
this desk tomorrow**. Ratchet steady at 6/8. **New today: that redesign is on
the GPU critical path** — `UB.10` is one of only two implemented GPU specs not
already settled, so the Review desk is itself one of the two things standing
between the builder and a dispatchable queue. Stated plainly because it is
uncomfortable and because nobody else will audit my deliverables.

**4. Organ liveness — all four fire on time; one of them does nothing.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| overseer | 6 h | 08-29 06:37 (45th, running now) | live |
| field watch | Mon 05:37 | 08-24 05:54 | live (next 08-31) |
| review | daily 06:37 | 08-29 06:37 | live (this) |
| builder | hourly | 08-29 06:07 | **fires on time; 90 slots, 0 work** |

`lost_iterations.log` still 0 bytes — correct, since no slot has *attempted* a
model since the fallback repair. Untested; do not trust it until it has fired.

**5. The cron collision fired again this morning, same second.** `overseer.log`
records `2026-08-29T06:37:03 audit start`; `review.log` records
`2026-08-29T06:37:03 review start`. The 45th audit and this Review are sharing a
git index right now. The code fix (`git commit -- <paths>` in every organ that
commits from a session) is still unimplemented four days after both organs
routed it independently; the owner's cron line is still unchanged. This commit
uses an explicit pathspec.

---

## 5. The honest paragraph, no numbers

Today the machine found out that it had been arguing about the wrong thing, and
it found out by looking at something other than itself. For days the whole
apparatus has been trained on a gate — whether it would open, when, at whose
expense, and who had mispredicted it — and the gate turned out to be downstream
of a much duller fact: the builder had run out of experiments to run. The shelf
was bare before the door was locked, and nobody checked the shelf, because we
have instruments for everything except the question "is there anything left to
do that would cost money?" That is the more honest shape of the last three
weeks, and it is worse than the story we had, because a locked door is somebody
else's fault and an empty shelf is ours. It also carries the first genuinely
actionable instruction this desk has issued since Tuesday: the work that makes
next week's free compute spendable requires no permission, no hardware and no
decision from anyone — it is writing the next experiment, which is the thing
this project is actually for. The week's most important step was made by
nobody, again. But the most concerning drift is no longer that we watch
ourselves instead of building him; it is that when we did finally look outward,
the thing we found had been sitting in plain sight in a file we all read daily,
and four organs walked past it because each was busy checking the last one's
arithmetic.

---

## REWRITTEN / STRENGTHENED

**None — DAILY mode. Part 2 runs Sunday 08-30**, together with the anatomy
audit, the completeness audit, and the `UB.10` arm redesign owed that day.

---

## FOR THE BUILDER — ordered

**B0. Commit `experiments/tests/sm_03_nose_reports_occluded.py`.** Fifth day,
eighth organ asking. 32 KB, the only runnable claim spec for *smell*, the only
spec of 187 whose implementation git has never seen, one `git clean` from gone.
Commit it with its state stated honestly — implementation only, pilot never
completed (`/data/sm03_pilot_seed90.json.log` is 0 bytes), gates not frozen —
and **do not dispatch it.** I again decline to sweep it into a Review commit
(`c0afded` bans exactly that).

**B1. REFILL THE GPU QUEUE — this is now the top of the board.** §2: all 17
runnable GPU-cost specs are unimplemented (7), settled (7), parked (2) or
unfrozen (1). Implement ONE unimplemented GPU spec end to end with its controls
— `T2.09` (Noisy-TV, kills ICM alone), `T3.06` (ablate curiosity; that
commitment has 12 specs and 1 pass), `T2.19`, `T2.11`, `T2.14` — chosen by `run
next` and the frontier, not by that order. **It needs no GPU, no meter and no
owner decision**, and it is the only thing that makes W35's 30 free hours
spendable. Three weeks of "dispatch early" advice has been aimed at a shelf that
was empty.

**B2. Build the instrument that would have caught this: queue depth.** Add to
`coverage.py` (beside `goal_citations()`) a count of specs that are
**runnable AND implemented AND not settled AND not parked AND tracked in git**,
split by cost class, with a shrink-only baseline. Today's GPU value is **0** and
no instrument in the repo can say so. This subsumes carried item B9 below (the
registry×index join for untracked implementations) — an untracked implementation
is just one way to have zero queue depth. Pair with the 36th audit's
`gpu.py:274` fix (`--untracked-files=no` hides a brand-new spec file from the
push guard).

**B3. Move the meter rule into `docs/LESSONS.md`, and extend it.** Carried from
yesterday, now with a second half. *"Do not model the meter"* lives only in
`ladder_prompt.md`, which the auditing organs never open — five falsified
attempts to price organ-hours in nine days. Add: **"and do not model the line
either — compute it."** The line is `PACE_FLOOR + ((PACE_CAP-PACE_FLOOR)*elapsed
+ 99)/100`, exactly 0.3869 pts/h, zero variance; the 44th audit fitted it by
least squares and derived a 243-hour wait 40 minutes before the gap closed to 1
point (§3). Generalised form: *a quantity you can read out of the source is not
a quantity to estimate — and a lesson written in one organ's prompt is not a
lesson the system has learned.* Yours to write; I am not putting words in your
file.

**B4. `git commit -- <paths>` in every organ that commits from a session.**
Carried, unimplemented, and the collision fired again at 06:37:03 today (§4.5).
`git add <named-paths>` does **not** protect you — `git commit` writes the whole
index. `ladder_loop.sh:166` has the correct form; `overseer.sh`, `review.sh`,
`field_watch.sh` delegate to their agent session where nothing enforces it.
Better fix: a pre-commit hook refusing a commit touching another organ's output
file (`PROGRESS*.md` → Review, `OVERSIGHT.md` → overseer, `FIELD_WATCH*.md` →
scout) unless that organ authored it.

**B5. Fix the quantifier in `experiments/champions.py` and its fixture.**
Carried verbatim from the 43rd and 44th audits, still owed. `all(v == "NOT_RUN"
…)` discharges a seat when any arena spec has run — including the incumbent's
own arm. Change to `not challenger_runs`, count a run only when its ledger status
is PASS or FAIL and its registry `COVERS:` kind is not `fixture`/`rule`/`sensor`
(import `coverage.py`'s parser). Fix `_fixture()` in the same commit. Do not
repair this by editing `CHAMPIONS.md`.

**B6. Verify the background-liveness rule the ledger already carries.** Carried
from the 30th audit, third victim now (`SM.03`'s pilot). Before writing
`iteration end rc=0`, check any claimed live background work has a live pid and
a non-empty declared artefact; log a distinct nonzero outcome naming the orphan.
The rule is in `LESSONS.md`; nothing implements it.

**B7. Baseline the ratchets that count only one class.** Carried (40th audit).
`champions.py:449` counts `ARENA-MISSING` only — deleting 13 phantom ids yields
a *perfect* ratchet while five seats go permanently unfalsifiable.
`decisions.py` has the same shape with `NO-DEFAULT`. Precedent: `coverage.py`
had this exposure and `T0.21 P2` closed it.

**B8. Make `overruns: []` mean what the audits read it as meaning.** Carried.
`2026-W31` records 37.46 h against a 30.0 ceiling with an empty overruns list.
Standing invariant + a known-answer property in `T0.21`.

**B9. Report skip streaks.** Carried. Ninety have now passed and no instrument
can say so; the number cached in `ladder_prompt.md` was understating by 24 until
I removed it this morning.

**B10. Verify the fallback repair rather than trusting it.** Carried. Your first
slot refuses on Fable if it lands before 08-31 04:59; expect `LIMITED on fable —
falling back to opus`. If instead you see `rc=1` in three seconds with
`lost_iterations.log` still 0 bytes, say so loudly.

**B11. `SH.02` implementation** — tier 2, CPU_LONG, deps all PASS, no owner
gate, no GPU. Take it only after B1; it does not refill the queue.

---

## FOR THE OWNER — strategic forks only

**1. The GPU emergency was misdiagnosed, by me among others, and the fix is not
the one three days of escalation asked you for.** This desk and the overseer
have pressed `JACK_NO_PACE=1` as urgent to save W34's 29.69 free hours. Per §2
those hours were unspendable from Tuesday morning regardless of the gate: the
last dispatchable GPU spec was consumed **8.4 hours before the blackout began**,
and the shelf has been empty since. Setting `JACK_NO_PACE=1` today still costs
nothing and still buys a real window (the 90% stop is untouched, 17 points of
headroom), so it remains worth doing — but it would buy a commit and a build
unit, **not a dispatch.** I would rather correct the diagnosis than keep the
urgency that got your attention.

**2. The fork that replaces it: GPU-spec inventory is unowned, unmeasured, and
it is what actually cost 61 hours.** Jobs per GPU week ran **17 → 23 → 1** while
every document blamed uptime. No instrument counts dispatchable specs; the
builder's priorities have said "dispatch early" for three weeks without ever
asking whether there was anything to dispatch. **Recommendation: make queue
depth a first-class, ratcheted number (builder B2) and a standing builder duty —
an iteration that finds GPU queue depth at zero implements a GPU spec before it
does anything else.** I have put that in `ladder_prompt.md` as a priority, which
is operational and mine to set; making it a *rule* is constitutional and yours.

**3. The pace gate still deserves your ruling, on its own merits and no longer
as an emergency.** It has one call site — the builder, the only organ that writes
to the ledger — while three Opus organs share the meter ungated; its own source
comment diagnoses this nine lines before installing it; it has produced 90 dark
slots at a meter that never exceeded 73% against an owner rule of 90%. The 44th
audit appended **`D15`** (armed, due 09-05) proposing to pace three of the four
daily audits instead. My recommendation is unchanged — **suspend `pace_gate` and
let the 90% hard stop be the only limit** — but §2 means it is no longer the
thing standing between you and spent GPU hours, and I am ranking it below fork 2
accordingly.

**4. Eleven decisions default-fire on 2026-08-31, one date, one hour** (`D15` is
the twelfth, 09-05). `D1` costs 38 specs, `D10` 8, `D4` 8. Per the 40th audit,
on 09-01 there will be 11 OVERDUE rows covering 54 specs and `decisions --check`
will still **exit 0** — `overdue` is a row field, never a violation.
Recommendation unchanged: **answer `D1` and `D10`, or re-arm both past the W1
design** (owed by this desk 08-30).

**5. Unchanged, and still the one that matters most: build W1, do not patch W0.**
Four independent instruments say the world is too thin to be worth learning.
Design owed by this desk **2026-08-30**, together with the `UB.10` arm redesign
which §4.3 now shows is on the GPU critical path.

**6. The cron collision fired again this morning at the same second** (§4.5).
`37 */6` and `37 6 * * *` put the 45th audit and this Review in one git index.
Nothing was lost — this commit uses an explicit pathspec — but the code fix has
been routed by both organs for four days and is still unimplemented, and cron is
outside every organ's mandate. **Free fix: `37 3,9,15,21 * * *` for the
overseer.**
