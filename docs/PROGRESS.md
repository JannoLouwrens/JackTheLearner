# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-26 06:40 UTC — DAILY. Window: 2026-08-25 06:40 → 2026-08-26 06:40.**

*The one sentence: **the builder has been dark for eighteen consecutive hourly
slots, throttled by a spend meter that its own auditors are driving — measured,
not inferred: in the eighteen hours it was skipped, the un-throttled organs were
98.9% of the burn, and the throttle is on the only organ that produces
science.***

---

## 1. The numbers

**Ladder: 84/187 demonstrated (44.9%).** At window start: 84/181 (46.4%). At
the 08-21 Review: 83/169 (49.1%).

| | this window | previous |
|---|---|---|
| demonstrated | **84** | 84 |
| registered | **187** | 181 |
| rate | **44.9%** | 46.4% |
| net new PASS | **+0** | +0 |
| rework (attempt > 1) | **62.5%** (60/96) | 60.7% |
| ledger totals | 84 PASS / 9 FAIL / 3 VOID | — |

**Runs in the window: four. All four were the same spec.** `T0.21` at 07:13
(PASS), 09:14 (FAIL), 09:16 (PASS), 10:14 (PASS) — the guard-of-the-guard that
audits `coverage.py`, re-stamping its own hash after each edit to the file it
audits. **Not one run in twenty-four hours tested a capability of Jack's.**

**Registry +6, demonstrated +0.** `SH.02` and `SM.03` (`f0cb81d`) and the `LG`
family — `LG.00/01/02/10` (`ed2d969`). Both commits were correct and both were
the top item on an auditor's page. Both are registrations.

**Goodhart check — and this window is the clean example.** Three ratchets went
green in it: `coverage` exit 2 → 0, `0 CLAIM-DEAD`; GOAL.md dangling citations
5 → 4; two champion seats `ARENA-MISSING` → `UNCONTESTED`. Every one of those
was discharged by **declaring a falsifiable claim**, not by Jack passing one.
*shelter/building*, *thermal (kills)* and *smell* all still read **`0 pass`**.
The rate fall (46.4% → 44.9%) is the honest direction of that: the denominator
is the truth. **14 of 23 commitments have a live claim spec and nothing
passing.**

**Last first-ever PASS: `NE.00`, 2026-08-24** (a `rule`-kind spec — not credited
as a claim by `coverage`). **Last *claim* PASS: `T3.01`, 2026-08-20 15:29 —
5.7 days ago.** Eight of the 84 PASS carry a claim marker. *(Correcting my own
08-21 line, which dated T3.01 at 08-21 01:28: that is the attempt-5 re-run's
timestamp, not the first PASS. The overseer's 08-20 was right.)*

**The frontier, recomputed live (not quoted).** `T2.01` (FAIL, 2.67σ against a
5σ bar) has a transitive block mass of **36** — up from 35 on 08-25 and still
the largest single blocker in the ladder. Behind it: `T4.04` 9, `T3.02` 9,
`T4.01` 8, `T2.16` 8, `NE.01` 8, `LC.03` 8. **69 of 187 specs carry at least one
non-PASS transitive dependency; 34 are runnable now.** T2.01 is settled behind
D1/D9, and none of the top seven waits on compute.

**GPU: W34 has spent 0.31 of 30 hours.** 29.69 free Kaggle hours expire at the
weekly reset on **2026-08-30**. W32 lost ~13.4 h unspent; W33 lost 22.1 h. This
would be the **fourth consecutive week**, and ~65 hours cumulative.

**Is the builder working on the frontier?** It cannot be. It has not run since
2026-08-25 12:23.

---

## 2. THE FINDING — the throttle is on the wrong side of a feedback loop

**What happened.** The builder ran six iterations on 08-25 (00:07 → 12:07), all
`rc=0`, and has run **zero** since. From 13:07 through 06:07 the log carries
**eighteen consecutive `PACING:` lines**. No commit has landed from the builder
since 10:16 on 08-25. Three consecutive overseer audits have returned
**DRIFTING** and named this; the 31st audit's item B1 was scheduled *on the
organ that was being skipped*, which is why it is still un-executed.

**The gate.** `pace_gate` draws a line from 25% at the weekly reset to 90% at
week's end and skips the builder above it. Right now: `week:all models` **52%**
at **29%** of the week, line **44%**. The gate is functioning exactly as
written.

**The premise is what fails.** `ladder_prompt.md` justified exempting the
auditors with *"they are ~18% of organ runs"* — a **run count standing in for a
spend**. Measured from the session transcripts, 2026-08-19 → 08-26:

| organ | sessions | requests | output tokens | cache-write | per-session out |
|---|---|---|---|---|---|
| builder (Fable) | 84 | 6,960 | 8,030,013 | 24,635,862 | **95,595** |
| overseer + review (Opus) | 23 | 1,857 | 2,181,741 | 7,552,501 | **94,858** |

The per-run cost is a **dead heat**. The auditors are ~21% of sessions and ~21%
of output tokens — but they run **Opus** where the builder runs **Fable**, so
their share of the *meter* is larger than their share of the tokens. And in the
blackout window itself:

| session | window | model | requests | output tokens |
|---|---|---|---|---|
| overseer 00:37 | 00:37–00:48 | opus | 99 | 146,526 |
| overseer 18:37 | 18:37–18:48 | opus | 65 | 118,553 |
| overseer 12:37 | 12:37–12:47 | opus | 91 | 98,428 |
| builder (dying tail) | 12:23 | fable | 6 | 4,047 |

**98.9% of the on-box spend during the builder's blackout was the organ auditing
the blackout.** That is not greed — it is a loop. Skipping the builder raises
the auditors' share of the burn, which pushes the meter further above the line,
which skips the builder again. Measured: over the twelve hours 13:07 → 01:07 the
meter rose **+14 points** while the allowance rose **+4**. **The builder lost ten
points of headroom in a day in which it did nothing**, and *not running is its
only lever.*

**The forecast, with its assumptions named.** The line rises 0.387 pts/h. Each
overseer audit costs ~4.5 all-models points and there are four a day (~0.75
pts/h). The deficit is 8 points and widening at ~0.36 pts/h. **The builder does
not run again this week under the current configuration.** The meter reaches the
90% hard stop — which silences *all four* organs — after roughly eight more
audits, around **2026-08-28**. The 29.69 Kaggle hours expire on 08-30, inside
that blackout. The eight armed owner decisions fire on 08-31, at the reset.

**One correction to the 32nd audit, on the evidence.** It attributed the burn to
"two interactive sessions in `/home/opc` (30.3 MB and 5.9 MB), live through the
blackout, carrying **both** Opus and Fable turns". I read both transcripts: their
last messages carrying a `usage` block are **2026-08-24T11:10** and
**2026-08-04T11:04**. Neither has had a model turn during the blackout. Their
recent mtimes are not model turns. The overseer was right that it was one of the
suspects and right to say so; it was wrong about who the other one was, and the
correct answer is that there was no other one.

**And `week:Fable` is not the builder's meter.** This has been stated as fact on
the builder's page three times. In the 18 hours when the only on-box spend was
three Opus audits and a 6-request Fable tail, `week:Fable` went **66% → 86%**.
Read straight off the log: one builder Fable iteration moves it ~1 point; one
overseer Opus audit moves it ~7. It stands at **88%** — under two points of room,
and less than one audit's worth. I am not going to guess at the mechanism; the
measurement stands on its own and the operational rule follows from it.

---

## 3. The second finding — a registered spec whose only copy is untracked

`SM.03` — *"The nose reports what the eye cannot"*, the successor spec that
took *smell* off the CLAIM-DEAD list — is registered in
`registry_expansion.py:2368` and **its ~710-line implementation is untracked in
the working tree**, 18 hours on. It was orphaned when the 12:07 iteration
reported a pilot as "pid 1552865, ~667 MB, healthy" that had already died with
its session: pid gone, `/data/sm03_pilot_seed90.json.log` **0 bytes**, no result
JSON. `scripts/launch_detached.sh` exists for precisely this and was not called.

Two audits declined to sweep it into an auditor's commit, correctly — `c0afded`
banned exactly that — and so do I. But the consequence is now a state nothing in
the system can see: **`run coverage` reports `SM.03 RUNNABLE` for a spec that
does not exist in git.** Every instrument reads the registry; none joins the
registry to the index. That is the same shape as the CLAIM-DEAD hole and the
prose-only dependencies — a fact true in the world and absent from the standard
the audits measure against. A guard for it is item 2 below.

---

## 4. Steering maintenance (Part 2.5) — done

**1. `scripts/ladder_prompt.md` — two fixes, both mine to make.**

- The meter section's exemption clause (*"~18% of organ runs… so they are not
  throttled"*) is replaced with the measured table above and named as a feedback
  loop. Added: a run of `PACING:` lines is not the same animal as one, and the
  builder must count and report a streak in its first paragraph — **a skip
  streak is the one fault this gate cannot report about itself, because the
  organ that would report it is the organ being skipped.**
- Added the `week:Fable` correction with the measured per-organ prices. The rule
  it protects (all-models is the gate) is unchanged; only its false premise is
  gone.
- The priority block's *"THE THREE CLAIM-DEAD COMMITMENTS ARE THE HIGHEST-VALUE
  CPU WORK"* was discharged on 08-25 and would have sent the next iteration to
  write a third successor spec. Re-pointed at what is actually live —
  implementation, and the orphaned `SM.03` file first — with the honest note that
  CLAIM-DEAD → RUNNABLE moves a commitment from *unmeasurable* to *unmeasured*.

**2. `docs/FIELD_WATCH.md` — nothing owed.** Unchanged since sweep wk4
(`474061d`, 08-24), consumed in full by the 08-25 Review. Next sweep Mon 08-31.

**3. Seat staleness — no new finding.** `champions --check` exits 0, ratchet
steady at 6/8 phantom arenas. Both DEFAULT seats have tracked resolutions:
*Learning core* is PENDING D10 (armed, 08-31), *Vision encoder* is contested and
carries T3.01 as its first real defence. *Sensory fusion*'s `UB.10` arm redesign
is routed to this desk and lands with the W1 design on 08-30.

**4. Organ liveness — and a distinction this check did not previously draw.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| overseer | 6 h | 08-26 06:37 | live |
| field watch | Mon | 08-24 05:54 | live |
| review | daily | 08-26 06:37 | live (this) |
| builder | hourly | 08-26 06:07 | **fires on time; has done no work in 18 h** |

The builder passes the liveness check. It reports, punctually, once an hour,
that it is not working. *"Silence is never success"* was written for an organ
that stops speaking; this one is speaking. **A liveness check that reads the
clock and not the output cannot tell a working organ from a skipped one** — the
`PACE-STREAK` marker in item 4 below is the cheapest fix.

---

## 5. The honest paragraph

We are not closer to Jack, and this time the reason is not that the science went
badly — the science did not happen. The machine spent the day auditing itself
with great care, and everything it found was true: the registry gained real
falsifiers, the constitution's own citations resolve further than they did, the
sense that had no claim has one again. But a claim is a promise to measure, and
nothing was measured. The single most important step toward Jack this window was
giving smell and shelter falsifiable successors after both had been honestly
retired — that is the ladder doing the hardest thing it knows how to do, which
is to keep a commitment alive after killing the test that carried it. The most
concerning drift is that the organ which builds is the only one being rationed,
by a meter its auditors are filling, and the arithmetic says it stays rationed
until the week ends and then everything stops together. What worries me is not
the outage; outages end. It is the shape: we have built four organs to watch one
organ work, and when the budget got tight the system protected the watching. The
watchers are honest, they found this themselves, and one of them recommended
cutting itself first — and none of that changes the fact that a creature does
not get built by anyone here except the builder. Free compute is about to expire
for the fourth week running while a page of perfectly true findings accumulates
about why it wasn't spent.

---

## FOR THE BUILDER — ordered

**B1. Commit `experiments/tests/sm_03_nose_reports_occluded.py` before anything
else.** It is a registered spec whose only copy is unversioned. Then re-launch
its pilot with `scripts/launch_detached.sh` — the loop's `timeout 25m claude -p`
reaps a harness-tracked background task with the session, which is the third
recurrence of this class and the second time it has cost a whole unit.

**B2. Make the orphan class unrepeatable — a registry×index join.** No
instrument in the repo checks that a registered, non-PARKED spec's
implementation is *tracked in git*. Add it to `coverage.py` beside
`goal_citations()` (a new untracked implementation exits 2, shrink-only
baseline), and add the known-answer battery to `T0.21` as `P12` the way `P11`
was added for the parked leak. This is the "machine better than found" item for
whichever iteration wakes first — it is CPU-free and it closes a hole that three
separate findings have now come through.

**B3. Report skip streaks.** `ladder_loop.sh` should count consecutive `PACING:`
slots and, past 6, emit a `PACE-STREAK n` line and touch a marker file the
overseer reads in its liveness section. A pace skip is not a fault; eighteen of
them is, and today nothing in the system could say so without a human asking.

**B4. Propose — do not implement unilaterally — a builder RESERVATION in
`lib_usage.sh`.** The fix is not to throttle the auditors symmetrically: that
would have silenced the overseer that found this, and the auditors are the
machinery that catches drift. The fix is that the pace line should be drawn
against *the pool the builder may actually have* — reserve a fixed share of the
week for the builder and pace the auditors against the remainder. Write it up
with the arithmetic and route it to the owner as a decision; **the level of
spend is the owner's call and neither of us may lower a gate to reach our own
work.**

**B5. `SH.02` implementation** — tier 2, CPU_LONG, dependencies all PASS, no
owner gate, no GPU. It is the only claim-kind work on the board that needs
nothing from anybody. If the builder gets one hour this week, this is the hour.

---

## FOR THE OWNER — strategic forks only

**1. The builder will not run again this week unless you act, and I recommend
both levers, not one.** The arithmetic: the deficit is 8 points and widens at
~0.36 pts/h; the builder's only lever is not running, which makes it worse.

- **(a) Cut the overseer to 12-hourly** (`37 */12`). It recommended this itself,
  against its own interest, and my attribution confirms it is the largest
  non-builder draw. But it is **not sufficient**: two audits a day is ~0.375
  pts/h against a line rising 0.387 — that stops the divergence and repays none
  of the deficit.
- **(b) Also set `JACK_NO_PACE=1` (or `.usage-resumed`) until 08-30**, so the
  builder can reach the 29.69 Kaggle hours before they expire. The 90% hard stop
  stays in force either way; this only removes the *pacing* line, which is the
  thing that has now cost 18 hours and is forecast to cost the week.

I did not pull either lever. `.usage-resumed` prints *"RESUMED BY OWNER"* and
forging that signature is exactly the dishonesty this system exists to prevent;
cron is outside my mandate. **The one thing I would not do is nothing** — the
forecast blackout starting ~08-28 takes the auditors down with the builder, and
last time that happened it lasted 4.3 days.

**2. Which meter actually gates this account?** `week:Fable` is at **88%** and
is driven ~7:1 by the Opus auditors, not by the Fable builder. Every document
here says it "is not the gate". If that is right, it is a number nobody needs to
watch; if it is wrong, the system has under two points of room and does not know
it. This is the third time in three weeks that a meter's *meaning* — not its
value — has cost us days (08-21: the builder grounded itself on the wrong line;
08-24: "90%" was prose with no authoritative source). **My recommendation stands
from 08-21 and I am raising it again: make the tooling name its own gating
number and assert it, rather than leaving each organ to read a line and infer.**

**3. Eight decisions default-fire on 2026-08-31 — inside the forecast
blackout.** `D1` costs 38 specs, `D10` 8, `D4` 8. D1's written default describes
a four-arm bakeoff **that does not exist as a spec**, so firing it buys a ruling
and no motion on the project's largest blocker. Recommendation: **answer D1 and
D10 before 08-28, or re-arm both past the W1 design** (owed by this desk 08-30).
Letting them fire into a silent system is the worst of the three branches — it
spends the decision and gets nothing built.

**4. Unchanged from 08-25, and it is the one that matters most:** build **W1**,
do not patch W0. Four independent instruments say the world is too thin to be
worth learning. Design owed by this desk **2026-08-30**.
