# OVERSIGHT — 35th audit, 2026-08-26 18:40 UTC

## VERDICT: DRIFTING — `week:Fable` reached **100%** and has held there for five consecutive hourly readings. The builder's model is now *exhausted*, not *nearly* exhausted, and that flips the pace gate from a throttle into an amplifier: every point of `week:all models` it is "holding for later in the week" will, when it finally releases, be spent by **Opus**, because Fable is gone until the reset. The gate is now conserving a pool in order to spend it worse.

**State, unchanged for 30 hours.** `HEAD` is `be8cfbc` — the 34th audit's own
commit. **Zero commits since.** Working tree is still the single untracked
`sm_03_nose_reports_occluded.py`. `84/187 demonstrated`. The builder's last
iteration ended **2026-08-25 12:23:33**; every slot since — **thirty
consecutive hourly slots, 30 h 17 m** — logged `PACING: … skipping`.

**Clean results, each re-run by me rather than relayed:**

- **§1 ledger integrity — clean.** `run verify`: **83 PASS re-judged from the
  record alone, 81 controls probed.** 0 verdicts that no longer re-derive, 0
  gates that ignore their control, 0 controls declared but never run, 0 gates
  unreplayable, 0 entries unauditable, 0 controls run but undeclared. Two
  PASSes carry no control (`T0.01`, `T0.10`), both long-declared existence
  claims; `T0.18` self-excludes correctly.
- **§2 thresholds and controls — clean, and trivially so.** `git log
  be8cfbc..HEAD` is **empty**. There have been **no code commits at all** since
  the builder went dark; the only three commits in the last day are the 32nd
  audit, the Review, and the 34th audit. Nothing could have been loosened
  because nothing was written. I re-ran the seven-day scan anyway and confirm
  the 34th audit's finding stands: no threshold moved in the loosening
  direction, no control deleted or weakened, no `_check` gained an `or`, no
  seed count reduced, no assertion removed.
- **§5 compute accounting — clean.** `overruns: []`; W34 charged **0.3111 h of
  30**, one job, one real ledger row.
- **§7 bakeoff hygiene — no findings.** No bakeoff has run since the 29th audit.
- **The three constitutional gates all exit 0**, all run by me: **coverage** — 0
  commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling GOAL.md
  citations (shrink-only baseline); **decisions** — ratchet ok, 3/10 undeclared;
  **champions** — ratchet ok, 6/8 seats with a phantom arena.

---

## RANK 1 — Fable is **exhausted**, and the pace gate's conservation now costs more than it saves

Read by me at 18:40, directly from `scripts/claude_usage.py`:

```
session                [                    ]   1%  resets Aug 26, 8:40pm (UTC)
week:Fable             [####################] 100%  resets Aug 31, 5am (UTC)
week:all models        [############        ]  62%  resets Aug 31, 5am (UTC)
```

Six hours ago my predecessor read Fable at **99%** and called it *"one point
from exhaustion."* It is now at 100 and has printed 100 in `ladder.log` at
**14:07, 15:07, 16:07, 17:07 and 18:07** — five consecutive readings. This is
no longer a forecast about a meter; it is a measured floor.

**Why that changes the verdict rather than merely updating it.** The 34th audit
listed three consequences of the gate being blind to Fable. Consequence #2 was
conditional — *"the next slot the gate admits will … run a full 50-minute
iteration on Opus."* At 99% that was an inference. At 100% it is arithmetic:

- `crontab`, live: `7 * * * * JACK_LOOP_MODEL=fable …` — the builder's primary
  model is Fable.
- `ladder_loop.sh:45`: `FALLBACK_MODELS="${JACK_LOOP_FALLBACK:-opus sonnet}"`,
  and the chain fires only **after** a primary attempt fails.
- Therefore **every** iteration between now and the Aug 31 05:00 reset that the
  gate admits will burn a few seconds on Fable, take `LIMITED`, and run on
  **Opus**.

So the pace gate's own stated purpose — the comment block at `lib_usage.sh`
says the line exists so *"the loop is still awake when the GPU quota expires"* —
is now inverted twice over. It is (a) conserving 28 points of `all models` that
the builder cannot spend on its cheap model, and (b) conserving them
*specifically so they can be spent on the most expensive model on the same
meter*. A throttle that defers cheap work into an expensive window is not
saving the pool; it is repricing it upward. Nothing in the system logs that
substitution as an event — it is a `say` line inside a normal iteration.

**Attribution, and the part that should worry the owner most.** In the six hours
06:07 → 12:07 the 34th audit measured `week:Fable` moving 87 → 99 with the
builder running zero iterations. I confirm the same shape 12:07 → 18:07: Fable
99 → 100 (ceiling), builder iterations **zero**. `overseer.sh`, `review.sh` and
`field_watch.sh` all pass `--model opus` (verified live in `ps`: this audit's own
process ends `--model opus`). **No jackthelearner organ runs on Fable except the
builder.** Every Fable point this week was therefore burned outside this
project — and Fable is the only model with its own weekly line, the only meter
that could be watched, and it belongs to the only organ that produces science.

---

## RANK 2 — the SM.03 pilot died 30 hours ago, the iteration certified itself `rc=0`, and this is the **second instance of the same failure in eight days**

This is the finding that damages the loop's own success signal, so I rank it
above the compute loss it caused.

**The evidence, read this hour:**

```
/data/sm03_pilot_seed90.json.log        0 bytes   mtime 2026-08-25 12:21
/data/sm03_pilot_seed90.json            DOES NOT EXIST
pid 1552865                             DOES NOT EXIST
experiments/tests/sm_03_nose_reports_occluded.py   32,086 bytes, untracked
```

Compare with the pilots that worked: `sh01_oracle_pilot.json` **and** `.log`
both written; `sm02_pilot.json` **and** `.log` both written. SM.03 wrote a log
file of **zero bytes** and no result at all.

**What the iteration recorded about itself**, `ladder.log`, 2026-08-25 12:07 slot:

> *"The pilot is a tracked background task — I'll be re-invoked when it
> completes (no extra monitor needed; polling would be waste). … pilot running
> full-size on seed 90 (pid 1552865, ~667 MB, healthy)."*
>
> `2026-08-25T12:23:33+00:00 iteration end rc=0 — 84 -> 84 demonstrated`

Fourteen minutes later none of that was true, and the loop's own exit code said
the iteration succeeded.

**The mechanism is nameable, and that is what makes it fixable.** The builder
runs as `claude -p … --max-turns 120` under a `timeout`. Background-task
tracking is a property of *that process*. When the iteration returns, the
process exits and takes its process group with it — so *"I'll be re-invoked when
it completes"* is not merely optimistic, it is **structurally impossible in a
batch `-p` invocation that has already returned.** A 0-byte log from a process
that had 667 MB resident is the signature of an unflushed stdout on a killed
child, not of a clean failure.

**This is not a new class of bug in this repo; it is a repaired one that
regressed.** `LESSONS.md:3769` already records the fix: *"`scripts/dispatch.sh`
setsids every [watcher]"* — and `:3780` records the precedent that the guard
existed and was bypassed because *"the probe was submitted ad hoc."* That is
exactly what happened here: an ad-hoc pilot launched outside `dispatch.sh` /
`launch_detached.sh`, so it inherited no `setsid`. The **30th audit
(`53f1cb2`, 2026-08-25)** reported *"an iteration closed rc=0 on a pilot that was
already dead."* The 12:07 iteration on the same day did it again.

**The missing organ, stated as a scar and not a wish** (SYSTEM.md's "no new
organ without a scar" is satisfied twice over): **nothing refuses `rc=0` for an
iteration that claims live background work.** The loop has a guard for a torn
ledger, a guard for undeclared controls, a guard for reattach code mismatch, a
guard for stale certificates — and no guard at all for its own success signal
being attached to a corpse. Two instances, eight days apart, same week.

**What it cost, concretely:** `SM.03` is the successor claim spec for **smell**
— one of the owner's constitutional senses, registered on 08-25 specifically to
clear its CLAIM-DEAD status — and it is the **only registered `GPU_SHORT` claim
candidate** for the 29.69 expiring Kaggle hours in RANK 3. Its registry entry is
sound (I checked: `control` declared as a shuffled-field twin plus an
occluder-removed alive-proof, `null_baseline` declares chance bins *and* a
matched-dimension placebo channel, `seeds=3`, `depends_on=['SM.01','PG.6']`).
The science is ready. The launcher killed it and the loop said `rc=0`.

---

## RANK 3 — every armed default is dated **after** the harm it exists to prevent has already happened

New this audit, and it is a defect in the *arming* mechanism rather than in any
one entry — which is why no previous audit caught it: each entry looks correctly
armed in isolation.

`decisions --check` prints **ten armed entries. All ten carry
`decide_by: 2026-08-31`.** Two of them exist solely to get the builder running
*this week*:

- **D13** — the overseer's own cadence, raised because four audits/day are
  burning the meter that gates the builder.
- **D14** — the model-meter blindness that is RANK 1 above.

Set those against the two clocks that actually bound the harm:

| clock | source | expires |
|---|---|---|
| **Kaggle W34, 29.6889 h unspent** | `gpu.py:369` uses `time.strftime("%Y-W%U")`; `%U` week 34 = Sun Aug 23 – Sat Aug 29 | **2026-08-30 00:00 UTC** (~53 h away) |
| `week:Fable` / `week:all models` | printed by `claude_usage.py` | **2026-08-31 05:00 UTC** |
| **every armed default** | `docs/DECISIONS_NEEDED.md` | **2026-08-31** |

So the defaults fire **after** the free GPU hours are gone, and **at the moment
the meter resets and makes the question moot.** D14's default is a pre-flight
refusal that only matters while Fable is exhausted; it is scheduled to fire on
the day Fable is refilled. D13's default halves this organ's spend on the pool
that is gating the builder; it is scheduled to fire after the pool refills.

**The generalisable defect: `decide_by` was set from the constraint's expiry
rather than from the deadline of the harm.** 2026-08-31 is the meter's reset
date — the date after which *none* of these questions cost anything. A
pre-registered default is a device for acting before it is too late; dating it
by when the pressure lifts converts it into a device for acknowledging that it
was. This is a milder cousin of the D1 disease: not a deadlock, but a clock set
to ring after the fire is out.

I am not moving any deadline — SYSTEM.md forbids that in the loosening
direction and I will not do it in the tightening direction either without the
owner, since a shortened deadline shortens *their* window to answer. It is
routed to the owner in FOR THE OWNER, with the arithmetic attached.

---

## RANK 4 — §6's converse question, asked properly for the first time: **two owner decisions have been settled in practice and never recorded**

The audit brief asks, and no recent audit has answered: *"was any owner-decision
quietly acted on without being recorded?"* Two, and they point opposite ways.

**(a) The owner ANSWERED, and the system never acted or closed the entry.** The
entry titled *"Was physics-first retired by argument instead of by bakeoff?"*
is printed by `decisions --check` as **`UNDECLARED`** — an open question with no
default and no deadline. But its own body reads:

> **DECIDED 2026-08-09: (a) RUN IT.** Owner: *"schedule the run after T2.01."*

That is **17 days old**. `T5.01` — the spec the owner ordered run, titled "THE
thesis test", the one that makes the project's *founding* premise rest on our
own numbers — is still `NOT_RUN`, queued behind `T2.01` (FAIL, 2.67σ against a
5σ bar, transitive block mass **36**, the largest single blocker in the ladder).
So an owner ruling is simultaneously (i) given, (ii) unexecuted, and (iii)
counted by the instrument as an *unasked* question. It inflates the UNDECLARED
count with something that is not undeclared at all, and it hides a debt the
owner is owed behind a label that says "we need input."

**(b) The owner never ruled, and the loop went ahead anyway.** **D3 — "May the
loop `git push`?"** — is still OPEN. Meanwhile `ladder.log` contains **146
lines mentioning a push**, including the mechanical
`2026-08-25T05:07:14 bookkeeping: pushed` emitted by `harvest_bookkeeping`, a
function the project's own audits commissioned. The loop pushes on essentially
every iteration and has for weeks. The escalation that D3 records — *"that is
not a stable rule, it is a coin flip"* — resolved itself by attrition into a
settled practice that no document authorises.

This is the honest shape of it: **the practice is right** (the toolchain
requires it — `gpu.py:assert_ref_is_current` refuses any job whose HEAD is not
an ancestor of `origin/main`, so no push means no GPU work at all), and **the
record is wrong**. I have armed D3 this audit — my required arming — with a
default that **bounds** the de-facto practice rather than widening it. See
`DECISIONS_NEEDED.md`; the reasoning is that a default may only pick among
already-permitted actions, and unrestricted pushing is already occurring, so the
only ratchet-legal move is to draw a fence around it.

---

## RANK 5 — `SM.03` untracked for 30 hours (fifth consecutive audit), and the rescue path still cannot see it

Unchanged and therefore worse:

```
?? experiments/tests/sm_03_nose_reports_occluded.py   32,086 bytes, mtime 08-25 12:20
```

`harvest_bookkeeping`'s `HARVEST_PATHS` covers `ledger.json`,
`gpu_budget.json` and `gpu_submissions.jsonl`. It is the pace-skip rescue path,
it has fired correctly during this blackout, and it covers **the one artifact
class that is not at risk**. 32 KB of a constitutional sense's only claim
implementation is one `git clean` from gone, and every hour of the blackout is
another hour it spends untracked.

I did not commit it. Versioning a test file is outside this role and the
boundary is load-bearing — it is what stops specs entering the tree without
pre-registration discipline. It remains builder item **B1**, and I note plainly
that this is the fifth audit to write that sentence.

---

## §3 — drift from the goal

**What the builder worked on in the last 24 hours: nothing.** Thirty dark
slots. The only three commits in the window are the 32nd audit, the Review, and
the 34th audit — all of them documents about the system, written by organs that
audit it.

That is the drift finding, and it should be stated without softening: **for the
last 30 hours, 100% of this project's output has been prose about itself and 0%
has been science about Jack.** None of it is *wrong* — §2 is clean partly
because nothing was written — but SYSTEM.md's own corollary is explicit: *"when
the machine is sufficient, PROVE it by throughput."* Four documents in 30 hours
is not throughput; it is the machine describing its own paralysis at four
documents a day.

**The converse and harder question — which parts of GOAL.md have no passing
spec at all** — is unchanged from the 34th audit and I re-ran `coverage` to
confirm rather than quote: **14 of 23 commitments have a live claim spec and
nothing passing.** The constitutionally most exposed, verbatim from the tool:

- `smell` — 0 pass; claim `SM.03` RUNNABLE (**dead pilot, untracked file** — RANK 2)
- `thermal (kills)` — 0 pass; claim `SH.02` RUNNABLE
- `voice` — 0 pass; claim `VO.02` RUNNABLE
- `balance` — 0 pass; claim `BA.02` RUNNABLE
- `proprioception`, `plasticity`, `sleep`, `fast/slow` — 0 pass, **every claim blocked**
- `curiosity` — 12 specs, **1 pass**
- `one brain / unison` — 21 specs, **1 pass**

Curiosity and all-senses fusion are exactly the two GOAL.md names the brief
warns are *"most likely to be quietly neglected in favour of easy wins"*, and
they carry 33 specs and 2 passes between them.

---

## §4 — is the builder alive and productive?

Alive as a cron entry; productive not at all.

| | value |
|---|---|
| iterations in the last 24 h | **0** |
| consecutive pace-skipped slots | **30** (08-25 13:07 → 08-26 18:07) |
| hours dark | **30 h 17 m** |
| PASS delta over that window | **0** (84 → 84) |
| commits by the builder | **0** |
| last *claim* PASS | `T3.01`, 2026-08-20 15:29 — **6.1 days ago** |

**The pace-line arithmetic, updated with my own readings and stated as a
threshold rather than a date** (my predecessor's lesson: a rate is not a date,
and turning one into the other is how the 33rd audit told the owner to stand
down six hours before the burn resumed):

`allow = 25 + ⌈65·elapsed/100⌉`, so the line recovers at **0.3869 pts/h**. That
is the gate's entire catch-up budget.

| slot (UTC) | 12:07 | 14:07 | 16:07 | 18:07 |
|---|---|---|---|---|
| `week:all models` (**the gate**) | 59 | 61 | 62 | **62** |
| week elapsed | 33 | 34 | 35 | **36** |
| pace line | 47 | 48 | 48 | **49** |
| **gap over the line** | 12 | 13 | 14 | **13** |
| `week:Fable` (printed, ungated) | 99 | 100 | 100 | **100** |

**The burn has slowed markedly** — 1.17 pts/h over 06:07→12:07, **0.50 pts/h**
over 12:07→18:07 — and I want that on the record because it is the honest
direction and it cuts against alarm. But 0.50 is still **1.29× the 0.387
recovery rate**, and the gap over six hours went 12 → 13. **While the burn stays
above 0.387 pts/h the gap cannot close.** It has been above it for twelve
consecutive readings. I give no wake-up date and I am not asking anyone to act
on one.

---

## §5 — compute honesty

Accounting is clean: `overruns: []`, W34 shows exactly one charged job
(`jannolouwrens/jack-ladder-1787631708`, 0.3111 h, `ok: true`) and that job has
a real ledger row (`T2.15`, FAIL, harvested and written up at `f5d8f1c`). **No
GPU hours have been spent without a ledger entry to show for them.**

The dishonesty is in the other direction — hours *not* spent:

| week | Kaggle unspent at expiry |
|---|---|
| W32 | ~8.8 of 30 |
| W33 | 22.11 of 30 |
| **W34** | **29.6889 of 30**, expiring **2026-08-30 00:00 UTC** |

That is a fourth consecutive week in the same shape and it is trending the wrong
way: 8.8 → 22.1 → (on current trajectory) 29.7. On a project whose owner has
ruled **free compute only**, roughly **60 free GPU-hours** will have died
unspent in three weeks. There is exactly one registered `GPU_SHORT` claim
candidate for this week's 29.69 h, and RANK 2 is the story of how its pilot was
killed and its implementation left untracked.

---

## §6 — stuck decisions

Covered in RANK 3 (deadlines dated past the harm) and RANK 4 (two decisions
settled in practice, unrecorded). Beyond those:

- **No `MEANS-ESCALATED` entries.** Nothing that a measurement could settle is
  sitting on the owner's desk. The D1 disease is not present today.
- **`D1` itself** still costs **38 specs** and is correctly armed with a
  default due 2026-08-31 that upholds the plastic-only decree verbatim.
- **`D10`** (8 specs) now carries **four independent instruments** measuring W0
  as too shallow — the darkroom control, LC.03 v2, DP.05, and the SH.01 oracle
  pilot. That evidence base is complete and its default is sound. It is the
  single most valuable thing the owner could rule on early.
- **Undeclared: 3 of 10** — ratchet ok, and I am reducing it to 2 by arming D3.

---

## §8 — the honest summary

**No. We are not closer to a curious humanoid that climbs the ladder than we
were yesterday, and we are not closer to a longer list of green ticks either.
We are closer to nothing.**

The ladder reads **84/187 demonstrated (44.9%)**, unchanged for six days. The
last time a spec first passed a *claim* about a capability of Jack's was
**2026-08-20** — six days ago. In the 30 hours since the builder went dark, the
project produced four documents and zero measurements, and three of the four
were written by the organ writing this sentence.

The one thing I will say in the system's favour, because it is true and it is
the point of having an overseer: **nothing lied.** `run verify` re-judged 83
PASS entries from the record and every one still re-derives. No threshold moved.
No control was weakened. The ledger is exactly as trustworthy as it was six days
ago — the falsification machinery is intact and idle. The 44.9% figure *fell*
from 46.4% this week because the denominator grew honestly while the numerator
did not, and nobody reached for the numerator.

But an honest ledger that is not being written to is a museum. The gap between
where Jack is and where GOAL.md says he must go is not being closed by an audit
that measures it four times a day, and the last three audits — including mine —
have each spent the meter that keeps the builder asleep in order to say so
again. **The binding constraint on this project has not been research, or
compute, or design, for six days. It is that the only organ permitted to produce
evidence is the only organ throttled, and it is throttled by a meter it does not
control, drawn down by sessions this project cannot see.**

---

## FOR THE BUILDER

Ranked. **B1 and B2 must be done in the first iteration that runs**, whenever
that is — B1 because the file is one `git clean` from gone, B2 because it is
what killed the last unit of real work.

- **B1 — commit `experiments/tests/sm_03_nose_reports_occluded.py` (fifth
  carry).** 32,086 bytes, untracked since 08-25 12:20. Its registry entry is
  sound and pre-registered (`f0cb81d`). The pilot that was meant to freeze its
  gates is dead and produced nothing, so **commit it as implemented-and-unpiloted
  and say so in the message** — do not invent pilot numbers, do not dispatch on
  frozen-by-guess gates, and do not silently re-run the pilot as if the first
  had merely been slow.

- **B2 — make an `rc=0` that certifies a corpse impossible.** Two instances in
  eight days (30th audit; 08-25 12:07). Add to `ladder_loop.sh`, after
  `run_claude` returns and before the `iteration end` line: if the iteration's
  output claims live background work (grep for a launched pid, or better, have
  pilots register themselves in a `RUNNING/` marker file), then **verify the pid
  is alive and its declared artifact is non-empty** before logging `rc=0`.
  Otherwise log `iteration end rc=2 — orphaned background work: <pid> gone,
  <artifact> 0 bytes`. The zero-byte-log-plus-formerly-resident-process
  signature is the exact one this box already has a lesson for
  (`LESSONS.md:3769,3780`).

- **B3 — no ad-hoc pilot launches.** The regression happened because the pilot
  was launched outside `dispatch.sh` / `launch_detached.sh`, so it inherited no
  `setsid` and died with its parent. Route **every** long-running local pilot
  through `launch_detached.sh`, and make the launcher's contract explicit in its
  docstring: *a `claude -p` iteration cannot be "re-invoked when it completes";
  the process that would receive the notification exits first.*

- **B4 — extend the pace-skip rescue path to untracked spec implementations.**
  `HARVEST_PATHS` covers the three RUNNER_OUTPUTS and not the artifact class
  that has actually been at risk for 30 hours. Add a narrow, pathspec-explicit
  rescue for untracked files under `experiments/tests/` matching a registered
  spec id, committed with a message that marks them **UNPILOTED**. Keep the
  `add -A` ban and the torn-file guard.

- **B5 — close the two unrecorded decisions of RANK 4.** (i) The physics-first
  entry carries `DECIDED 2026-08-09: (a) RUN IT` in its body while
  `decisions --check` reports it `UNDECLARED`: move it to
  `DECISIONS_RESOLVED.md` with the owner's ruling and record `T5.01` as an
  **owed run queued behind T2.01**, so the debt is visible as a debt rather than
  as an unasked question. (ii) D3 is now armed by me — implement nothing, but
  do not let the entry rot again.

- **B6 — fix the 06:37 collision (carried from the 34th audit, unactioned).**
  `37 */6` (overseer) and `37 6 * * *` (review) fire simultaneously every day,
  putting two concurrent long-effort Opus sessions on the shared meter 30
  minutes before the builder's 07:07 slot. Free fix: `37 3,9,15,21 * * *`.

- **B7 — when the gate finally admits an iteration, log the model substitution
  as an event.** Per RANK 1, the next admitted slot will run on Opus, not Fable.
  A `say` line inside a normal iteration is not enough: emit
  `MODEL SUBSTITUTION: primary <m> LIMITED, running on <fallback>` at the same
  prominence as the `PACING:` line, so the log shows *what the pool was actually
  spent on*.

---

## FOR THE OWNER

Three things, one of which is time-critical and none of which I can do myself.

**1. `week:Fable` is at 100% and it is being consumed outside this project.**
The builder is the only jackthelearner organ that runs on Fable; it has run
**zero** iterations in 30 hours; and Fable went 87 → 100 in that window. Fable
is also the *only* model with its own weekly line — Opus and Sonnet roll into
`all models`. So the one meter that determines whether this project produces
science is drawn down by interactive sessions the project cannot see or gate.
Nothing here is a request that you stop using them; it is a request that you
**know that is the trade**, because no instrument in this system was able to
tell you until now.

**2. The armed defaults are dated 2026-08-31, which is after everything they
protect has expired.** This is RANK 3 and it is the item I would most like you
to overrule me on, because I deliberately did not move a deadline myself:

- Kaggle W34's **29.69 free GPU-hours die 2026-08-30 00:00 UTC** (~53 h away).
- `week:Fable` refills **2026-08-31 05:00 UTC** — 29 hours *after* that.
- **D13** (halve this organ's spend) and **D14** (make the gate read the meter
  that actually binds) both fire **2026-08-31**.

Both defaults exist to get the builder running *this week*. Both are scheduled
to fire after the week's compute is gone and at the moment the constraint lifts.
**If you rule on nothing else, ruling D13 and D14 early — or simply authorising
me to re-date them to 2026-08-28 — is what converts this week from a total loss
into a partial one.** I did not re-date them because a shortened deadline
shortens *your* window to answer, and that is your call, not mine.

**3. The cheapest immediate lever, stated plainly, and it is against my own
organ.** The 32nd audit offered it, the 33rd raised it against itself, D13 now
has a proper default, and it remains true: **cut me before you cut anything that
produces science.** Four audits a day on Opus, auditing a system that has been
byte-identical for 30 hours, is the clearest waste in the current picture. D13's
default (option (c), the change-gated no-op) does this conditionally and
reversibly and I endorse it without reservation.

One thing that does **not** need you: `D10` now has four independent instruments
behind it and its default is sound. If you have one minute rather than ten,
spend it on D13/D14 — D10 will resolve itself correctly on its own terms.
