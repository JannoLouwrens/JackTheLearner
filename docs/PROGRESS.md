# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-20 06:45 UTC — DAILY. Window: 2026-08-19 06:40 → 2026-08-20 06:40.
First Review in six days: the five before it were refused at the usage gate.**

*The one sentence: **the loop woke from a 4.3-day blackout and had its single
best day of real science — taste became the first constitutional commitment
this project has ever closed with a claim — and then spent its last three
iterations waiting on background processes that had already died in silence.***

---

## 0. The gap this page has to declare first

**The system was dark 2026-08-15 00:07 → 2026-08-19 07:31 UTC.** Every organ
was refused at the 90% weekly-usage stop: **~103 builder fires, 5 Review fires,
and the Monday field-watch sweep**, all logged with a reason, none executed.
Zero ledger rows were recorded in those 4.3 days. The 08-14 Review predicted
this collision to within about twelve hours and it happened exactly as
described, including the part that cost the most: **Kaggle's W32 hours expired
unspent on 08-16 because the loop was not awake to submit them.**

That is the correct reading of every number below. This is not a 24-hour window
inside a steady run; it is the first 24 hours after a long stop, and both the
good and the bad in it are shaped by that.

---

## 1. The numbers

**Ladder: 82/169 demonstrated (48.5%).** Last Review (08-14): 79/169 (46.7%).

| | this window (24 h) | 08-14 window | the dark days (08-15→08-19) |
|---|---|---|---|
| spec runs recorded | **52** | 27 | **0** |
| PASS / FAIL / VOID / ERROR / BLOCKED | 45 / 2 / 1 / 2 / 2 | 23 / 2 / 2 / 0 / 0 | — |
| **first-ever PASSes** | **2** (T2.06, TA.02) | 1 | 0 |
| net new demonstrated | **+2** (80 → 82) | +1 | 0 |
| registry growth | **0** (169, flat) | +3 | 0 |
| builder iterations | **24 fired, 24 rc=0** | 24 fired, 21 rc=0 | 0 of ~103 |
| commits | **42** | ~20 | 0 |

**Goodhart check: the rate rose 46.7% → 48.5%, and it is half an artefact.**
The registry did not move at all — 169 specs for six days — so every point of
the rise is the runner, none of it is the ladder growing underneath. That is
the *good* direction of the two failure modes this check watches for, but a
flat registry across 42 commits is its own signal: the builder registered no
new spec this window. It was running and repairing, not extending. One window
of that is healthy; a second would mean the research queue has stopped feeding
the ladder.

**Rework: 50 of 82 passing specs took more than one attempt (61.0%),** up from
52.4% on 08-14. Most of this window's contribution is honest — the stale-cert
recovery chain re-ran 17 specs and TA.02 needed a second attempt after a
harvest bug — but the number is now high enough to watch. It is the fraction of
our certificates that did not hold the first time.

**Zero-pass constitutional commitments: 15, down from 17.** This is the number
that matters most on this page and **it moved for the first time by a claim
rather than by registering a spec.**

**Compute.** Credits `week:all models` **46%**, Fable 48%, session 15%, reset
Aug 24 — healthy. Kaggle W33: **~3.7 h charged of 30, ~26 h expiring Sunday
2026-08-23.** The pressure has inverted since 08-14: credits are abundant, GPU
hours are abundant *and perishable*, and the binding constraint is now having
something worth submitting.

**The frontier, recomputed (not quoted).** Transitive unpassed-descendant mass:

| rank | spec | status | ready? | mass |
|---|---|---|---|---|
| 1 | **T2.01** Locomotion beats a random policy | FAIL | **READY** | **36** |
| 2= | T4.04 / T3.02 / T3.01 (ablations) | NOT_RUN | T3.01 READY | 9 |
| 2= | **LC.03** learning-core screening | **VOID + STALE** | **READY** | 8 |
| 5 | UB.10 fusion bakeoff | NOT_RUN | READY | 5 |

30 specs are runnable now; **57 are blocked behind something unpassed.** T2.01
alone accounts for 36 of them and has been FAIL since 08-12 — eight days,
unchanged, correctly declined for re-running (a re-run is a seed lottery
against a fixed 5σ bar; the measured value is 2.67σ on a converged curve). It
is a science problem and nobody is working on it, because D1 — where control
lives — has been open on the owner's desk for **eleven days** and its
`RECOMMENDED` option is the one the PLASTIC-ONLY decree forbids.

---

## 2. What the window actually produced

**Two first-ever PASSes, and one of them is the best result in weeks.**

- **TA.02 — one-trial conditioned taste aversion. PASS.** Pooled avoidance
  0.983, the Garcia dissociation clean, the DQN null eating 196–218 toxic meals
  at zero discrimination over 150 lives. **Taste is the first zero-pass
  constitutional commitment ever closed by a claim spec.** Every previous
  "Jack got a sense" headline — voice, balance, cold, smell — was a *fixture or
  sensor* passing, the apparatus rather than the capability. This one is the
  capability. It is the single most important step toward Jack this week.
- **T2.06 PASS** (P100, 1069 s billed, 0.30 h of W33).

**Three negatives, all routed correctly.** XL.01 FAILed twice — the carried
diary bought nothing on fresh worlds 3–5, refuted first by its own control and
then on the claim. SH.01 was parked at z=1.03 rather than spending 80 CPU-min
on a guaranteed VOID. The instrument is still refusing things, which is the
only reason its greens are worth anything.

**The machine got harder, twice, both from overseer findings.** The stale-PASS
cascade closed by the detector's own testimony (10 stale → 5+1, not one a
PASS); and the reattach-laundering hole is now mechanised shut — a
`JACK_REUSE_KERNEL` harvest can no longer stamp a certificate with code that
did not run, because `submit()` records the sha of the exact kernel pushed and
a mismatch is refused at zero billing. T0.24 is up to 8 properties.

---

## 3. THE FINDING — three iterations spent waiting on dead processes

**Measured at 06:41 UTC, and it is still true as this is written:**

```
/data/sm02_geo_check.log   0 bytes   launched 04:13   no process
/data/sm02_geo_occ.log     0 bytes   launched 05:08   no process
/data/sm02_geo_vis.log     0 bytes   launched 05:08   no process
/data/sm02_learnability_{vis,occ}.json   UNCHANGED since 03:13 / 03:15
ps -eo cmd | grep -E 'sm02|geo|experiments\.'   →   nothing
```

Three launches, three empty logs, zero surviving processes, and the downstream
JSONs still holding the 03:xx **Euclidean** numbers the geodesic repair was
supposed to replace. The comparable check that *did* run took 230–317 s and
printed on completion; these have been "running" for 93 and 33 minutes.

The three iterations that ended at 04:15, 05:10 and 06:12 all reported some
version of *"waiting on the two CPU checks — I'll be notified when both have
written their ratio lines."* Two of them ran under four minutes. **The builder
was blocked on a notification that could never arrive**, and the loop's own
liveness logging shows `rc=0` for all three, so nothing anywhere reads as
wrong.

**This exact failure was diagnosed by the builder itself yesterday at 10:41**
(`a51686c`: *"live oracle-probe pid is 3963630 — first two launches died at
import"*). A `setsid`-detached script launched from a `/data` cwd dies at
import before writing a byte; the launch returns 0; a poll on file *content*
waits forever. The lesson was learned, written into a commit message, and not
turned into a check — which is `LESSONS.md:4042` almost verbatim: *a scar
recorded in prose is prose; only a check binds the next author.*

**Also uncommitted:** `experiments/sm02_learnability_check.py` (untracked), the
sm_02 geodesic edit (+137 lines), the LESSONS entry, and the pilot's GPU
charge. **Five iterations of work sitting outside git**, in a project whose own
lessons file says a detached recorder stamps the tree it finds.

I have put all of this at the top of `scripts/ladder_prompt.md`, above the
priority order, because the builder fires at 07:07 and would otherwise wait a
fourth time.

---

## 4. Steering maintenance (Part 2.5)

**Four fixes to `scripts/ladder_prompt.md`, one of them expensive:**

1. **NEW §"Read this before you wait on anything"** — the dead-process finding
   above, with the relaunch procedure (`chdir` + `sys.path` pinned *inside* the
   script; `sleep 15; wc -c <log>` before believing a launch; `pgrep -f`, never
   the pid `setsid` returned).
2. **The expensive one: `§0` still ordered "LC.03 IS IN FLIGHT. DO NOT RUN IT
   AND DO NOT RELAUNCH IT"** — a Review directive from 08-14 naming `pid
   2536994`. That run **landed VOID on 2026-08-14 07:36** and the spec has
   since gone stale by `IMPL_DEPS`. **This desk's own order forbade the
   project's second-largest unblock for six days after it stopped being true.**
   Replaced with the live state and an instruction to read the VOID's reason
   before relaunching.
3. **`§2`'s credit/Kaggle section was entirely spent** — it warned about a
   collision "this weekend", quoted a 71% meter that reset on 08-19, and a
   Kaggle W32 deadline of 08-16. Replaced with the live meters and the
   inversion: the constraint is no longer the ability to submit, it is having
   something worth submitting.
4. **A cached count** — "`run coverage` still reads 17 and it has not moved" —
   on the very page that forbids cached counts. It is 15. Replaced with the
   instruction to read the tool.

**FIELD_WATCH.md: unchanged, nothing to consume.** Last real sweep 2026-08-12
(week 3), consumed by the 08-13 Review; the file was last committed 08-14
02:31, before the last Review read it. **The Monday 08-17 sweep was refused at
96% usage** — so week 4 does not exist and the scout has been silent for eight
days against a seven-day cadence. Logged with a reason, under 2× cadence, so
not a liveness finding — but a lost week, and the next sweep is 08-24.

**Seat staleness (`CHAMPIONS.md`, rule 4): one seat flagged, two cells
corrected.** The **Learning core** seat is FLAGGED STALE — a DEFAULT champion
that has never been contested, whose screening round has been VOID for six days
with nothing scheduled, and whose cell still read "IN FLIGHT since 2026-08-13".
Corrected to the ledger's actual state; **no seat changed hands.** The **Smell**
cell said `SM.02` "is runnable today"; it is now implemented, its first pilot
returned non-learning on all six arms, and the repair is uncommitted — recorded.

**Organ liveness: nothing silent, but the record is worse than it looks.**
Builder hourly, 24/24 this window. Overseer 6-hourly, all four fires present
(12:43, 18:46, 00:37, 06:37). `tmp_reaper` 04:13. `lost_iterations.log` empty.
Review and field watch: covered above — **every organ that was silent, was
silent because the usage gate refused it, and every refusal is logged.** The
gate works as designed. The design is the problem, and it is on the owner's
desk below for the second time.

---

## 5. The honest paragraph

We are closer to a creature, and I can point at the reason without reaching:
for the first time, one of the owner's own commitments stopped being a promise
and became a demonstrated capability. Taste is not a fixture that reports
numbers; it is Jack eating something that made him ill and then refusing it,
with the dissociation that tells us it is taste and not merely memory, and with
a null that shows a competent learner without the mechanism never gets there.
That is the shape everything else on this ladder has been aiming at and mostly
missing. Against it, the drift is the same one the last three pages have named
and it is getting older rather than better: the senses accumulate while the body
stays unproven. He can smell, taste, hear, feel cold and feel himself falling,
and he still cannot be shown to walk — and the reason is not compute and not
effort but a question about where control lives that has sat unanswered long
enough for the ladder to grow a whole wing behind it. The subtler worry is the
one this window exposed by accident: an organ that is stalled looks exactly like
an organ that is working, because the only thing anything checks is whether the
process exited cleanly, and a loop waiting patiently for a corpse exits very
cleanly indeed. We have built a system that cannot lie to us about its results.
We have not yet built one that cannot lie to us about whether it is doing
anything.

---

## FOR THE BUILDER — ordered

1. **STOP WAITING. The SM.02 geodesic CPU checks are dead** (evidence in §3 and
   at the top of `ladder_prompt.md`). Relaunch them with the repo path pinned
   inside the script, and *verify the artifact at 15 s* before you treat a
   launch as successful. **Never end an iteration on "waiting" without a
   liveness proof** — `pgrep -f` returning a pid AND the log growing.
2. **Commit the five iterations of uncommitted work** before any dispatch:
   `sm02_learnability_check.py` (untracked), the sm_02 geodesic edit, the
   LESSONS corollary, the pilot's GPU charge.
3. **Turn the scar into a check.** You have now lost time to
   dies-at-import-in-a-detached-launch twice in twenty hours, and written it
   down both times. Put a `launch_detached()` helper in the harness that pins
   `chdir`/`sys.path`, sleeps, asserts the log is non-empty, and returns the
   `pgrep`-found pid — or the third occurrence is already scheduled.
4. **LC.03 is unblocked, stale, and yours.** My own 08-14 order was what stopped
   it; that order is withdrawn. Read why attempt 1 VOIDed before relaunching —
   a second VOID for the same reason is the waste, not the run.
5. **~26 Kaggle hours expire Sunday 08-23 and credits are healthy.** Unlike
   last weekend you can actually spend them. Do not manufacture work to fill
   them; do make sure SM.02's repaired pilot and anything else genuinely ready
   is dispatched well before Sunday rather than at it.

## FOR THE OWNER — strategic forks only

1. **The 90% usage gate cost 4.3 days, ~103 builder iterations, all five
   Reviews, one field-watch sweep and a full expiring Kaggle allocation — and
   this is the second time I have raised it.** The failure shape is still
   wrong: the gate takes the *auditors* down with the builder, so the moment
   the project most needs oversight is the moment oversight is guaranteed
   offline, and nothing can even report that the loop is dark. **Recommendation:
   a small reserved slice for the three audit organs**, so that a builder
   blackout is observed rather than silent. Cheap, and it converts a total
   outage into a degraded one.
2. **D1 — where control lives — is eleven days open and it is now the single
   largest thing standing between this project and a Jack that moves.** T2.01
   blocks 36 specs, is correctly refusing to be re-run, and cannot be fixed by
   compute. Its `RECOMMENDED` option (frozen trunk + head) is barred by the
   PLASTIC-ONLY decree, so the decision is *unanswerable as posed* and has been
   for eleven days. **Recommendation: either narrow the decree's scope to admit
   a frozen control trunk into the arena, or strike that option and let the
   remaining arms run.** Both are decisions; continuing is not.
3. **An organ can stall indefinitely and every instrument we own will report
   `rc=0`.** §3 is the first observed instance: three iterations, exit code
   zero, nothing produced, nothing flagged. This is adjacent to the 08-13
   finding that no organ reads another's exit code, but it is worse — the exit
   code was *correct*. **Recommendation: a productivity heartbeat** (an
   iteration that commits nothing and moves no ledger row twice running is a
   finding), owned by the overseer. I have not implemented it; it touches organ
   scripts, which are outside this desk's jurisdiction.

---

*Part 2 skipped per DAILY mode. No threshold moved, no control softened, no
spec file touched, no ledger entry edited. Queued for Sunday's FULL run:
T0.13's 20+ attempts; SM.01's ungated intermittency shortfall; PS.03's
self-reported "one physics measurement, not three"; BA.01's control passing by
0.008; and the 61% rework rate against the oldest certificates.*
