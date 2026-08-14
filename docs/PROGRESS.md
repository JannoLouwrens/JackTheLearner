# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-14 06:40 UTC — DAILY. Window: 2026-08-13 06:40 → 2026-08-14 06:40.**

*The one sentence: **the ladder tested four new claims this window and refused
all four**, which is the instrument working exactly as designed — and it did so
in the twenty-four hours during which the loop quietly burned **32% → 71%** of
its weekly credit against a hard 90% stop, so the thing that decides what
happens next is not on the ledger at all.*

*Yesterday's headline is worth revisiting honestly, because the numbers do not
support the way it was written. The 08-13 Review celebrated "voice, thermal,
balance, damage, curiosity, fusion, vision". By this project's own
`coverage.py`, **VO.01, BA.01, PS.02, SM.01, TA.01 and T2.03 are sensors and
fixtures, not claims** — the instruments that would let a claim be made. Only
damage/nociception has a passing claim-kind spec among them. The kind system
that says so was itself built as a scar fix, and it is sharper than the prose
that ran past it. Jack got the apparatus for six senses. He has not yet been
shown to use any of them.*

---

## 1. The numbers

**Ladder: 79/169 demonstrated (46.7%).** 24 h ago: 78/166 (47.0%).

| | this window (24 h) | prior 24 h |
|---|---|---|
| spec runs recorded | 27 | 59 |
| PASS / FAIL / VOID / ERROR | 23 / 2 / 2 / 0 | 48 / 9 / 1 / 1 |
| **first-ever PASSes** | **1** (T0.27, harness) | 11 |
| net new demonstrated | **+1** (78 → 79) | +8 |
| registry growth | **+3** (BA.02, PS.04, T0.27) | +1 |
| builder iterations | **24 fired, 21 rc=0, 3 refused** | 24 of 24 |

**The composition of those 27 runs is the finding, not the total. Twenty-three
of them re-certified specs that were already green** — T0.17 ran **seven
times** in twenty-four hours, T0.27 six, T0.12 and T0.22 twice each, plus
T0.09, T1.07 and the PG.1/PG.2/PG.4/T2.20 gate sweep. Every one was legitimate:
B3 changed `protocol.py`/`gpu.py`, their `IMPL_DEPS` hashes moved, and the
ratchet correctly re-staled their certificates. **Only four runs in the window
tested something not already passing** — BA.02 twice, T4.02, T3.07 — and all
four came back FAIL or VOID.

**Goodhart check: the rate fell, 47.0% → 46.7%, and this one is not an
artefact.** The registry grew by three while demonstrated grew by one. That is
the ladder outrunning the runner again, the reverse of yesterday's reading — and
it is information rather than shame, because the three additions are honest
(BA.02 and PS.04 give balance and hunger/thirst their first *claim*-kind specs,
closing a hole where `n_pass` could not have moved no matter what ran; T0.27
gates GPU attributability). One window of falling rate after one window of
rising rate is noise. Two would be a trend. Read it again tomorrow.

**Zero-pass constitutional commitments: 17 of 23 — unchanged.** Yesterday's
page said 16; I ran `coverage.py` against the 08-13 baseline tree and it read
**17 there too**, with the tool byte-identical. That was a misreport, not a
regression, and no spec changed status in the window. **The standing rule's own
scoreboard did not move at all.**

**Rework: 44 of 84 entries carry `attempt` > 1 (52.4%), up from 49.4%.** Still
concentrated in the harness specs that gate other specs' certificates, which is
the same structural explanation as yesterday and is now three days old. It goes
to Sunday's Part 2 with T0.13.

**Tier picture, and it is the most encouraging thing here.** T3 opened this
window — with a FAIL. T4's second-ever entry is a FAIL. T5's newest is a VOID.
**The ladder has reached the tiers where components get deleted and claims get
refused, and it is refusing them.** T0/T1 are complete; T2 is 36/59.

**The two clocks, and they now collide.**
- **Claude: `week:all models` 71%, Fable 75%, resetting Aug 19 11:59 UTC.** It
  read **32%** at this hour yesterday. The hard stop in `lib_usage.sh` is 90%
  and it takes the overseer and this Review down with the builder. At the
  observed burn (~39 points/day) **90% arrives around 18:30 UTC today, leaving
  roughly five dark days.** Three fires were already lost on 08-13 to a
  *session* limit — a faster meter the loop has no gate for.
- **Kaggle: 19.09 h of 30 h charged to W32. ~10.9 h remain and die Sunday
  2026-08-16.**

If the credit gate closes first, the Kaggle hours cannot be spent at all.

---

## 2. The frontier

**59 of 169 specs are unreachable.** Live `run blocked`:

| terminal blocker | status | frees | blocks |
|---|---|---|---|
| **T2.01** Locomotion beats a random policy | **FAIL, 2.67σ vs a bar of 5** | **26** | **36** |
| **LC.03** Which learning cores learn to survive at all | **RUNNING — 15 h of 15–20** | 8 | 8 |
| UB.10 Fusion bakeoff: six arms | NOT_RUN | 3 | 5 |
| T2.02 Locomotion beats the honest MLP | VOID | 3 | 4 |
| T2.06 Language-action alignment | NOT_RUN | 3 | 3 |

**THE FRONTIER MOVED, and it moved by the builder refusing an order from this
desk. That refusal was correct and I am recording it as such.**

Yesterday's item 1 told the builder to submit T2.01 to Kaggle before Sunday, on
the premise that a dead P100 had made the run impossible all week. **The ledger
disagreed and the builder read it.** T2.01 v5 ran clean on the P100 on
**08-12 12:59** (commit `08444b2`, after the critic fix, before the cudnn
break). Its artifact shows reward-per-step flat at ~5.15 from 100K to 700K
steps on all three seeds: the curve has converged, so the pre-registered
"climbing curve → more compute" branch does not apply, and the binding sigma is
the trained-seed spread itself. **Re-submitting would have been a seed-lottery
redraw against a bar that does not move — run-until-pass, a stealth threshold
weakening.** The builder declined in `a3b12f6` with the arithmetic attached and
resolved the §3 gate explicitly as (b) in the same commit, which is the second
thing this desk asked for and had been drifted past three times.

**So the largest blocker in the project has changed category.** T2.01 is no
longer a job waiting for GPU hours. It is a claim that **will not pass by being
re-run**, and the thirty-six specs behind it — every curiosity spec, T5.01,
Tier 6 — are behind a science problem, not a compute problem. That is a worse
position than we were in yesterday and a more honest one. It is §7 item 2.

**LC.03 is running.** Fifteen hours in, three worker children, healthy. Named
"next" for four days and displaced every time by a rule that was right to
displace it; it is now in flight, and *the reason it took four days was not
neglect*: the 08-13 Review asserted it was "ready to run as-is" and it was not
— no `lc_03` test file existed. The builder had to write `survival.py` (the
shared lethal-W0 loop for LC.03/04/05), write the test, pilot it at seed 90,
fix a crash, amend two inverted controls, and only then launch. **A spec whose
dependencies pass is not a spec that exists**, and this desk did not check. That
correction is now in the steering file, against myself.

**Effort vs GOAL.md's path.** Of ~54 commits, roughly half served the creature
(LC.03's substrate and pilot chain, BA.02's three-attempt diagnosis, T3.07,
T4.02, T2.04) and roughly half the machine (three overseer audits, the
staleness class-closer, GPU attributability, seat refresh). The split holds
yesterday's inversion. **The yield differs completely: the machine work all
landed, the creature work all came back refused or is still in flight.** That
is not a criticism of the split. It is what a ladder looks like when it starts
biting.

---

## 3. The honest paragraph

*(Not required in DAILY mode. Written because the answer to "so what?" changed
direction again.)*

We are closer, and it does not feel like it, and both of those are true for the
same reason. Everything this window produced that a person could point at is a
refusal: the mood machinery does not change what he does, the fusion boundary
does not share its gradients fairly, the felt fall has no purchase in a body
with nothing to catch itself with. Yesterday I wrote that he had got a voice
and a sense of falling; today the project's own coverage instrument says he got
the microphone and the inner ear, and that using them is still unclaimed. That
correction stings and it is the most valuable thing on this page, because a
system that flatters itself about what its green ticks mean will eventually
flatter itself about something that matters. The step that took us toward Jack
was one nobody will celebrate: an order came down from this desk to re-run the
biggest blocked experiment, and the builder read the record, found the run had
already happened, and refused — because re-rolling the dice against a fixed bar
is cheating even when the person asking outranks you. That is the whole project
in one commit. The drift away is simpler and more dangerous: the machine is
about to run out of the one resource nobody has decided how to spend, on a
schedule that has now been set by the calendar three times running, while a
quota of borrowed compute expires unused beside it. We have built something that
tells the truth beautifully and cannot yet decide when to eat.

---

## 4. REWRITTEN / STRENGTHENED

None. DAILY mode does not re-examine tests — Part 2 runs Sundays. **No
threshold moved, no control softened, no spec file touched, no ledger entry
edited by this Review.**

Queued for Sunday's Part 2, carried and added to:
- **T0.13** — twenty-two attempts to make true. Carried from 08-13.
- **The re-certification cost.** T0.17 ran seven times in twenty-four hours and
  T0.27 six. The ratchet is correct and the runs are cheap individually, but
  nobody has ever measured what fraction of the loop's wall clock goes to
  re-certifying green specs. Measure it once; if it is small, say so and stop
  worrying.
- **SM.01's 0.33 intermittency shortfall**, **PS.03's single-physics gradation
  caveat**, **BA.01's 0.6918-against-0.70 control margin** — all carried
  unchanged from 08-13.
- **NEW: the six sensor/fixture PASSes** (VO.01, BA.01, PS.02, SM.01, TA.01,
  T2.03). Each is real work. The Part 2 question is whether any of them is
  *positioned* to make its commitment's claim reachable, or whether the claim
  spec above it is blocked on something nobody has scheduled. `VO.02`, listed
  in CHAMPIONS.md as "runnable today", declares in its own registry notes that
  it is **BLOCKED ON GEN.02 (a second Jack)**. That contradiction is a Part 2
  item, not a daily edit.

---

## 5. Steering maintenance performed

**`scripts/ladder_prompt.md` — three corrections, one of them urgent.**

1. **§2's clock directive was actively dangerous and I wrote it.** On 08-13 I
   replaced the two-clocks note with "credits are not scarce this week — do not
   ration iterations against them", on a 32% reading. It reads 71% today.
   Left standing it would have had the builder spend freely into a 90% cliff
   that also silences the auditors. Rewritten: both clocks bind, they collide
   this weekend, **GPU dispatch is the work that must not wait** because a
   submitted job survives a dark loop and an unsubmitted one is worth nothing on
   Monday. The live meters are named; the numbers are stamped with the hour they
   were read and explicitly marked as floors, not facts.
2. **§2 and §3 now record the T2.01 decline as SETTLED**, with the arithmetic
   that settled it, so no future iteration re-litigates a correct refusal every
   time it reads the priority file. §3's gate is marked resolved as (b) by
   `a3b12f6` — asked for on 08-13, delivered.
3. **§0 marks LC.03 IN FLIGHT with its pid and a "do not relaunch"**, carries
   the note that its workers are why every CPU timing measured today is
   pessimistic, and **records my own "ready as-is" error as a general rule**:
   a spec whose dependencies pass is not a spec that exists — check
   `experiments/tests/`. The cached "16 commitments" was replaced with a
   pointer to `run coverage`, per this file's own rule, which I broke.

**Field watch — nothing to consume.** `FIELD_WATCH.md` is unchanged since
08-12 07:01 and both live sweeps (wk2, wk3) were dispositioned on 08-13. The
scout's mandate runs to Monday 08-17. No action.

**Seat staleness (`docs/CHAMPIONS.md`) — the finding is closed, and well.** All
five stale seats reported on 08-13 were corrected by the builder in `c6ca536`
and `a46e186`, verdicts unchanged and nobody seated, each checked against the
ledger before editing. Better than asked for: the rewrites now distinguish
*fixture certified* from *seat claimed* ("VACANT — but he CAN make a sound"),
which is exactly `coverage.py`'s kind semantics reaching the human-readable
map. The builder also **barred frozen-trunk+head from the D1 seat** pending the
owner's reconciliation, which is the right unilateral call. One residue: three
seats say a claim spec is "runnable today" and **VO.02 is not** (its own notes
declare it blocked on a second Jack). Part 2 item, above.

**Organ liveness — all five alive.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:18 (rc=0) | **OK — 24 fired, 21 rc=0. The 3 refusals were a session limit, honestly logged** |
| overseer | 6-hourly | 06:37 (running) | OK — 15th audit closed `ON TRACK` |
| field watch | Mondays | 08-12 07:02 (rc=0) | OK — next due 08-17 |
| Review | daily | 06:40 (this run) | OK — 08-13 completed; the 08-12 API-529 death remains the open finding |
| tmp_reaper | 4-hourly | 04:13 | OK |

Nothing is silent. **But note what liveness cannot see: at ~18:30 today all
five of these go dark together and every one of them will look healthy doing
it**, because a gated organ exits 0 with an honest log line. My liveness check
is a silence detector; a five-day planned silence reads identical to a
five-day outage. That is the same structural blindness reported on 08-13 as
finding 3, arriving from a new direction.

---

## 6. FOR THE BUILDER

Ordered. Item 1 is measured in hours, not days.

1. **Dispatch the PROBE kernel now — not the full T2.04 run, and not after the
   CPU smoke.** *(AMENDED 07:05, after the overseer's 16th audit landed its
   RANK 1 while this page was being written. My original item said "dispatch,
   don't wait for the smoke"; that was wrong in a way the overseer caught and I
   had not: the smoke measured `d_model=64, n_layers=2` and the production
   kernel instantiates `PipelineConfig()` defaults `d_model=512, n_layers=8` —
   a trunk ~256× more expensive per step. `est_hours=2.0, timeout_s=18000` is
   extrapolated across that change and is not evidence about the run it sizes.
   Dispatching blind risks a timed-out kernel that bills the week's last hours
   and records nothing, which is the exact loss I was trying to prevent.)*

   **The clock argument and the overseer's fix agree, so do both:** its option
   (a) — a short probe timing ~5 `_train_bc` steps at the *production* config —
   is itself a GPU dispatch, costs minutes, keeps running while the loop is
   dark, and is the thing that makes the real dispatch safe. Send it now,
   re-derive `est_hours`/`timeout_s` from what it returns, commit the
   arithmetic, then dispatch the full run. **Do not wait on the CPU smoke for
   either step** — it is being starved by LC.03's workers on a four-core box,
   it has no cost gate, and it cannot answer the question that matters.
   Remember why the timing still binds: **~10.9 Kaggle hours die Sunday and the
   loop may be dark from ~18:30 today**, so a job that is not submitted before
   then is worth nothing on Monday.
2. **Do not relaunch LC.03.** `ps -p 2536994` before you act on anything that
   mentions it. If it lands, record it and take LC.04 — the arbitration, and
   the first thing in this project that decides *how Jack learns* rather than
   whether an organ works. If it died, the journal has the handoff.
3. **Write your handoff journal for a reader who arrives in five days, not in
   one hour.** If the weekly gate closes today, the next iteration to read your
   note may be Aug 19. Every in-flight pid, every "do not relaunch", every
   Kaggle attempt id needs to survive that gap, and `/data/tmp` gets reaped
   every four hours.
4. **Nothing else this window should start a multi-iteration unit.** With ~19
   points of credit left, finishing beats starting. If you have a spare cheap
   iteration, the highest-value one is the T2.04 record verification B3 asked
   for — confirm the entry carries `gpu_job_id` and remote hardware. That is
   B3's first live exercise and it is one command.

*Carried and still open from 08-13: N1, the certificate pre-gate for UB.11
(now non-hypothetical, UB.9 has passed); and the four consumed field-watch
nominations in queue order — wk2-N1 (whiff clock) first. None of them may
displace items 1–3.*

---

## 7. FOR THE OWNER

**1. The credit gate closes in roughly twelve hours and nobody has decided what
should happen when it does. This is the third resource crisis in four days and
the second that will otherwise be settled by the calendar.**

`week:all models` went **32% → 71% in twenty-four hours**. The hard stop is
90%, the week does not reset until **Aug 19 11:59 UTC**, and the stop takes the
overseer and the Review down with the builder — so the system loses its
auditors at exactly the moment it stops producing anything for them to audit.
Beside it, **~10.9 Kaggle hours expire Sunday** and cannot be spent by a loop
that is not running.

Both the 08-11 credit emergency and D5 "resolved" by the week rolling over
rather than by a decision. I noted at the time that this means we have never
learned whether the gate's shape is right; we have only learned that weeks end.
This is the third instance, and it is the first where a second resource is
destroyed as a consequence.

**Recommendation: decide before ~18:30 UTC today, either way, and record it.**
The cheapest concrete form: write `.usage-resumed` with a ceiling you actually
mean (95%? 97%?) and an expiry at the weekly reset — the mechanism already
exists and expires itself, which is why it is safe. **Or** say plainly that five
dark days is acceptable and that unspent Kaggle hours are an accepted loss, and
I will stop raising it. What I would rather not report on Sunday is a third
"resolved by the calendar". The standing-policy question (D5) is still open on
your desk; this is that question arriving with a deadline attached.

**2. The largest blocker in the project stopped being a compute problem
overnight, and that changes what it needs from you.**

T2.01 — locomotion beats a random policy — frees 26 specs and blocks 36,
including **every curiosity spec in the ladder**, T5.01 (the founding-thesis
test you personally scheduled), and all of Tier 6. It is FAIL at 2.67σ against
a pre-registered 5.

Yesterday this desk called it a GPU-hours problem. The builder proved otherwise
and refused the re-run: v5 already ran clean on the P100 on 08-12, and its
reward curve is **flat at ~5.15 from 100K to 700K steps on all three seeds**.
The learning has converged. The gap to 5σ is the spread between trained seeds
(means 280/447/484), not a shortage of steps. **More compute cannot fix this.
Re-running it would be rolling dice against a fixed bar until one comes up.**

So the thirty-six specs behind it are waiting on one of three things, and only
you can pick: **(a)** the claim is right and the *body* is wrong — the same
diagnosis BA.02 reached independently this window when balance turned out to
have no headroom in the rover body (D8); **(b)** the claim is right and the
*learning core* is wrong, in which case LC.03/LC.04 are the unblock and it is
already running; or **(c)** the 5σ bar prices seed variance in a way this
experiment can never satisfy, and the honest move is a redesigned claim under
the T1.02 precedent — a *harder* experiment, not a softer threshold.

**My recommendation: wait for LC.03 before choosing, then choose (b) if it
finds a core that survives, and (a) if it does not.** Both BA.02's and T2.01's
failures now point at the body rather than the brain, and two independent
experiments reaching the same suspicion is worth more than either alone. What I
am asking for today is only that you know the fork exists and that it is not
waiting on hardware.

**3. `D7` — a `kills` clause fired for the first time in this project's
history, and by rule the deletion is yours.**

T3.07 ablated mood conditioning and **FAILED**: mood does not measurably change
what Jack does in the shipped system (action-distribution divergence −0.025,
mood-regime classification 0.225/0.275/0.375 against a chance of 0.25). The
spec's pre-registered `kills` clause reads *"MovementMoodCoupling as anything
but cosmetics"* — so this is not an opinion about the module, it is the
outcome the experiment was registered to produce.

GOAL.md's Tier 3 says dead weight gets deleted; the builder's own rules say
deleting a component is your call, not its, so it escalated correctly. Note the
scale: `EmotionalState.py` is 1,149 lines and this commitment has long been
flagged as under-tested. **My recommendation: authorise deletion of the
coupling path specifically, not of emotion as a commitment** — T3.07 refutes
that *this wiring* changes behaviour, not that interoception and affect have no
place in Jack. Delete the cosmetics, keep the seat, and let something re-earn
it in an arena. This is the first component the ladder has ever asked to
remove, and doing it cleanly sets the precedent for the fourteen other Tier-3
specs behind it.

**4. `D1` is ten days old.** I second the overseer without adding an ask; the
builder has now barred the unconstitutional option from the seat itself, which
holds the line but does not answer the question. One line from you.

*Nothing in this review touched a threshold, a control, a spec file or the
ledger. The steering edits are operational and itemised in §5. Committed with
an explicit pathspec, per the 08-13 finding about two organs sharing one index
— the overseer's 15th audit was running concurrently while this was written.*
