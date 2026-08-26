# OVERSIGHT — 34th audit, 2026-08-26 12:37 UTC

## VERDICT: DRIFTING — and the specific thing that is broken is now **measured, not forecast**: the builder's own model quota (`week:Fable`) is at **99%**, one point from exhaustion, and **no gate in this system reads it.** The gate that is holding the builder asleep "to hold budget for later in the week" meters `week:all models` (59%). It is conserving a budget in a currency the builder's model will not be able to spend, and the two clocks that decide whether this week's free GPU hours live or die are **misaligned by 29 hours** — a fact published by the tools, requiring no extrapolation at all.

`HEAD` is still `b5499fe`. The working tree is still the single untracked
`sm_03_nose_reports_occluded.py`. `84/187 demonstrated` is unchanged. The
builder's last iteration ended **2026-08-25 12:23**; every slot since —
**twenty-four consecutive hourly slots, a full day** — logged `PACING: …
skipping`.

**I must open by correcting my predecessor, because the correction is the
finding.** Six hours ago the 33rd audit told the owner, in bold, *"No action is
needed from you on this"*, on the grounds that the burn had gone flat at 0.00
pts/h for five consecutive readings and the builder would wake unaided at
≈ Aug 27 05:07. It attached the right condition to that call — *"it holds only
while the foreign sessions stay quiet"* — and **that condition has failed.** The
burn resumed within the hour. This audit does not repeat my predecessor's error
of turning a rate into a date; it reports a **threshold**, a **measurement**
against it, and two **published timestamps**.

**Clean results, each re-run by me rather than relayed:**

- **§1 ledger integrity — clean.** `run verify`: **83 PASS re-judged from the
  record alone, 81 controls probed.** 0 verdicts that no longer re-derive, 0
  gates that ignore their control, 0 controls declared but never run, 0 gates
  unreplayable, 0 entries unauditable, 0 controls run but undeclared. Two PASSes
  carry no control (`T0.01`, `T0.10`) — both long-declared existence claims.
  `T0.18` self-excludes correctly.
- **§2 thresholds and controls — clean over seven days**, re-scanned
  independently. See §2 for the one diff I chased rather than trusted.
- **§5 compute accounting — clean.** `overruns: []`; W34 charged **0.3111 h of
  30**, one job, one real ledger row.
- **§7 bakeoff hygiene — no findings.** No bakeoff has run since the last audit.
- **The three constitutional gates all exit 0**, all run by me: **coverage** — 0
  commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling GOAL.md
  citations (shrink-only baseline); **decisions** — ratchet ok, 3/10 undeclared;
  **champions** — ratchet ok, 6/8 seats with a phantom arena.

---

## RANK 1 — the builder's own model is at **99%**, and not one gate in this system can see it

Read by me at 12:37, directly from `scripts/claude_usage.py`:

```
session                [###                 ]  19%  resets Aug 26, 3:39pm (UTC)
week:Fable             [################### ]  99%  resets Aug 31, 4:59am (UTC)
week:all models        [###########         ]  59%  resets Aug 31, 4:59am (UTC)
```

**The builder runs on Fable.** `crontab`, verified live:
`7 * * * * JACK_LOOP_MODEL=fable /home/opc/jackthelearner/scripts/ladder_loop.sh`.
**Both gates read `all models`.** `usage_gate` compares the 90% stop against
`_usage_pct()`, which is the all-models line. `pace_gate` does the same, and its
`week:Fable` read exists **only to print**, in a string that literally says
`(not the gate)` (`lib_usage.sh:112`).

So the loop's control surface is blind to the one meter that determines whether
it can run at all. Concretely, three things follow, none of them a forecast:

1. **The pace gate's stated purpose is not being served.** Its own comment
   block says the line exists so *"the loop is still awake when the GPU quota
   expires."* It is conserving `all models`, of which 41 points remain. The
   builder cannot spend those points on Fable; Fable has **1**.

2. **When the builder does wake, it will run on Opus.** `ladder_loop.sh:45`:
   `FALLBACK_MODELS="${JACK_LOOP_FALLBACK:-opus sonnet}"`. The chain fires
   *after* a primary attempt fails, so the sequence is: burn a slot, take
   `LIMITED on fable`, fall back, and run the full 50-minute iteration on
   **Opus** — the most expensive model on the shared meter the gate was built to
   protect. Nothing logs this as an event worth noticing; it is a `say` line in
   the middle of a normal iteration.

3. **The one meter with a separate weekly line is the one nobody guards.** Opus
   and Sonnet return *empty* from `--model` (I checked all three) — they have no
   distinct weekly line and roll into `all models`. Fable is the only model with
   its own ceiling, and it belongs to the only organ that produces science.

**Attribution, established rather than assumed.** In the six hours 06:07→12:07,
`week:Fable` moved **87 → 99 (+12)**. In that window the builder ran **zero**
iterations. `overseer.sh:47`, `review.sh:31` and `field_watch.sh:33` all default
to **opus**. No jackthelearner organ other than the builder runs on Fable.
**Therefore every one of those 12 points was burned outside this project** — the
two long-running sessions in `/home/opc` were both active (`68804e98` wrote at
12:09, `b76c8195` at 11:53).

**And I re-cleared the `/usage` probes by a different method than the 32nd
audit.** Its exoneration was correct: I opened the seven probe transcripts
written during this audit and **not one contains an `assistant` record**. The
`/usage` slash command is handled locally by the CLI with no model call, so the
gate's own meter reads cost nothing. That is a genuine no-finding and it holds.

---

## RANK 2 — the pace line has **no convergence guarantee**, and above a computable burn rate it stops being a throttle and becomes a permanent lockout

This is the structural defect underneath both the 32nd audit's alarm and the
33rd's retraction. Neither named it, and it is arithmetic, not opinion.

`allow = PACE_FLOOR + ((PACE_CAP − PACE_FLOOR)·elapsed + 99)/100`, with
`PACE_FLOOR=25`, `PACE_CAP=90`, `elapsed` in integer percent of a 168-hour week.

    elapsed rises at 100/168      = 0.5952 %/h
    allow   rises at 0.65 × that = 0.3869 points/h      ← the recovery rate

**0.387 pts/h is the entire budget the gate has for catching up.** If the shared
pool burns faster than that, the gap widens monotonically and there is **no
mechanism anywhere in the system that closes it** — not the gate, not a fallback,
not an alarm. The only exit is the weekly reset. A skipped slot is never
recovered; it is spent.

**Measured against that threshold, from `ladder.log` — the same instrument my
predecessor used:**

| slot (UTC) | 06:07 | 07:07 | 08:07 | 09:07 | 10:07 | 11:07 | 12:07 |
|---|---|---|---|---|---|---|---|
| `week:all models` (**the gate**) | 52 | 53 | 53 | 54 | 55 | 56 | **59** |
| week elapsed | 29 | 30 | 30 | 31 | 32 | 32 | **33** |
| pace line `25 + ⌈65·e/100⌉` | 44 | 45 | 45 | 46 | 46 | 46 | **47** |
| **gap over the line** | 8 | 8 | 8 | 8 | 9 | 10 | **12** |
| `week:Fable` (printed, ungated) | 87 | 88 | 89 | 91 | 93 | 94 | **99** |

- **Burn 06:07 → 12:07: `(59−52)/6 = 1.17 pts/h` — 3.0× the 0.387 recovery
  rate.** In the last hour alone it was 3 pts/h, or 7.8×.
- **The gap widened 8 → 12.** My predecessor's five flat readings were a lull
  inside a burn, not the end of one — and it said so itself in its sensitivity
  clause. The clause fired.
- **`week:Fable` burned at 2.0 pts/h** over the same window and 5 pts/h in the
  last hour.

**What I will state, and what I will not.** I will not give a wake-up date; that
is precisely the move my predecessor got wrong and its lesson is now in
`LESSONS.md`. What I will state is conditional and checkable: **while the burn
stays above 0.387 pts/h the gap cannot close, and the builder cannot wake before
the weekly reset.** The burn has been above that threshold for six consecutive
readings. For calibration only — *not as a prediction* — even if the burn
stopped dead at this instant, `pct=59` needs `elapsed ≥ 53` to proceed, which is
20 points away at 0.595 %/h. Six hours of measurement moved the *best possible*
case ~17 hours later. That is the direction of travel; the date is not the point
and I am not asking anyone to act on one.

**The generalisable defect:** a pacing line that meters a *shared* pool has a
recovery rate, and if it never compares its own recovery rate to the observed
burn it cannot tell a throttle from a lockout. It has now failed to tell them
apart for a full day. Lesson appended.

---

## RANK 3 — the two reset clocks are **29 hours out of phase**, and this finding needs no rate at all

The cleanest fact in this audit, because both numbers are *published* by the
tools rather than derived from anything:

| clock | source | expires |
|---|---|---|
| Kaggle W34 GPU quota (**29.69 h unspent**) | `%U` week 34 = Sun Aug 23 – Sat Aug 29 | **2026-08-30 00:00 UTC** |
| `week:Fable` / `week:all models` | printed by `claude_usage.py` | **2026-08-31 04:59 UTC** |

**28 h 59 m.** The free GPU hours die **before** the builder's quota comes back.
There is no ordering of events in which a reset-driven recovery saves this
week's compute: even a builder that wakes at the *instant* the meter resets
arrives a day and five hours after the hours are gone.

This is the fourth consecutive week in that shape — W32 lost ~13.4 h, W33 lost
22.1 h, W34 has spent 0.31 of 30 with one day of eligibility left. Roughly **65
free GPU-hours cumulative**, on a project whose owner has ruled free compute
only. The 32nd audit put this to the owner and the 33rd withdrew it on a
forecast. **The forecast was the wrong instrument; the phase difference was
always the right one, and it was true the whole time.**

---

## RANK 4 — `SM.03` has now been untracked for **24 hours** (fourth consecutive audit)

Unchanged, verified this hour, and therefore worse:

```
?? experiments/tests/sm_03_nose_reports_occluded.py   32,086 bytes, mtime 08-25 12:20
   /data/sm03_pilot_seed90.json.log                        0 bytes, mtime 08-25 12:21
```

The result JSON was never created; no pilot process exists. The 0-byte log with
a formerly-resident 667 MB process is the import-death signature this box has a
lesson for. The `12:07` iteration closed `rc=0` reporting *"pid 1552865, ~667 MB,
healthy"* and *"I'll be re-invoked when it completes"* — none of which was true
fourteen minutes later.

`SM.03` is the successor claim spec for **smell**, one of the owner's
constitutional senses, registered specifically to un-CLAIM-DEAD it, and the only
registered `GPU_SHORT` claim candidate for the 29.69 hours in RANK 3. It is one
`git clean` from gone. The pace-skip rescue path (`HARVEST_PATHS`) is scoped to
`ledger.json`, `gpu_budget.json`, `gpu_submissions.jsonl` — **it covers the one
artifact class that is not the problem.**

I did not commit it. Versioning a test file is outside this role's permissions
and the boundary is load-bearing: it is what stops specs entering the tree
without pre-registration discipline. It stays as builder item B1, and I note
plainly that this is the fourth audit to write that sentence.

---

## §2 — thresholds and controls, seven days: no findings

Re-scanned independently (`git log -p --since="7 days ago"` over `registry.py`,
`registry_expansion.py`, `experiments/tests/`, plus `git log -S` on every
threshold symbol on a removed line). **No threshold moved in the loosening
direction. No control was deleted or weakened. No `_check` gained an `or`. No
seed count was reduced. No assertion was removed.**

One diff I chased rather than waved through, because a line-grep flags it as a
removed `seeds=3`: in `ed2d969`, `DP.04`'s line
`budget=Budget.GPU_SHORT, seeds=3, depends_on=["DP.00", "VO.01"]` was reflowed
to two lines and became `depends_on=["DP.00", "VO.01", "LG.00"]`. **`seeds=3` is
unchanged; a dependency was *added*, tightening the graph** — the prose-only
constraint in DP.04's notes is now graph-visible. The 33rd audit's correction to
the 32nd (that `NE.01`'s `DELTA_T_NIGHT` 12 → 10 was a pre-run rig calibration
into the middle of a pre-declared band, with a recorded sweep, claim gate
untouched, and the spec FAILed anyway) stands on re-reading.

## §3 — drift from the goal

**The builder did no work in the audit window**, so there is nothing new to
trace and nothing that serves no GOAL.md sentence. Drift is not the problem
today; absence is.

The converse question is the one that matters and it has not moved. Of 24
constitutional commitments, **14 have live claim specs and nothing passing** —
`smell`, `voice`, `balance`, `thermal (kills)`, `shelter/building`,
`proprioception`, `plasticity`, `sleep`, `social/other agents`, `tool use`,
`touch/contact`, `death & retry`, `hunger/thirst`, and `fast/slow` (**8 declared
specs, 0 passing**). `curiosity`: 12 specs, **1 PASS**. `one brain / unison`: 21
specs, **1 PASS**. Those last two are what GOAL.md calls the thesis itself.

**Runnable *right now*, needing no decision and no unblocking:** `SM.03`
(smell), `SH.02` (thermal + shelter), `VO.02` (voice + social), `BA.02`
(balance), `XL.01` (death & retry), `LG.02` (social). Six constitutional
commitments have a runnable claim spec and a dark builder.

## §4 — is the builder alive and productive?

**Alive, correctly configured, and blocked by its own rule for a full day.**
Last iteration start `2026-08-25T12:07`, end `12:23:33`, `rc=0`. Every slot since
— **24 consecutive** — logged `PACING: … skipping`. No crash, no credit-exhaustion
marker, no repeated identical failure, no paused loop nobody resumed;
`lost_iterations.log` is 0 bytes and the cron entry is intact.

Iterations in the last 24 h: **0**. PASS delta: **0**. `84/187 demonstrated`
(44.9%, down from 49.1% as the registry grew 169 → 187 while PASS stood still —
the honest direction). Last PASS of any kind: `T0.21`, 08-25 10:14, a *guard*
re-stamp. Last **claim**-kind PASS: `T3.01`, **2026-08-20 — six days ago.**

## §5 — compute honesty

W34 charged **0.3111 h of 30**, a single job (`jack-ladder-1787631708`, `T2.15`),
which produced a real ledger row with a full write-up and a routed follow-up.
**No GPU hour this week was spent without a ledger entry.** `overruns: []`. The
W32 opening-balance gap (6.3849 h) is still carried in the over-stating
direction, which is the safe one. **The accounting is sound. The waste is not
mis-accounted spend — it is 29.69 hours that will expire uncharged**, per RANK 3.

## §6 — stuck decisions

Nine open decisions after this audit's addition, all with `decide_by:
2026-08-31`, **none overdue**, **none `MEANS-ESCALATED`**. No owner decision was
acted on without being recorded.

**On the three `UNDECLARED` entries — I re-verified rather than relayed, because
an audit calling a ratchet alarm "false" is exactly how a ratchet leaks.** All
three are false positives and none is armable:

- `DECISIONS_NEEDED.md:273` reads
  ``## ~~D3 — May the loop `git push`?~~ **ANSWERED: YES (owner, 2026-08-10)**``.
  It trips only because `_SETTLED` (`decisions.py:99`) matches
  `RESOLVED|off your desk|BY THE CALENDAR` and not `ANSWERED`.
- The two title-keyed entries (`:408`, `:558`) carry owner rulings dated
  2026-08-09 in their bodies under headers still reading `(OPEN, …)`, and
  **cannot be armed at all**: `_DECIDE` (`decisions.py:82`) forbids spaces in an
  id.

So **no genuine arming was available from the existing three**, and arming an
already-answered decision would be worse than the alarm. **This audit instead
arms a real new one — `D14`** (RANK 1/2/3: which meter gates the builder, and
what happens when its own model is exhausted). It is not manufactured: it is a
live resourcing fork with no default and no clock anywhere in the repo, which is
the `D1` shape, and the standing duty is to give it both. Its default is a
**narrowing** — it makes the loop refuse *more* than the 90% stop and moves no
threshold. See `DECISIONS_NEEDED.md`.

I also re-read `D13`'s armed default after my predecessor's `B8` warning about
`_FIELD` overwriting wrapped continuations: it parses correctly and its printed
default is *Option (c) — the CHANGE-GATED NO-OP*, which is what its author
intended. The parser footgun is real and still unfixed; I checked my own `D14`
block the same way, by reading the printed default rather than trusting exit 0.

## §7 — bakeoff hygiene

No bakeoff has run since the last audit. Re-read rather than assumed: `PS.01/J`
stands as a **VOID**, not laundered into a verdict; `PS.01/J2` declares its
2.66σ margin and its `screen` rationale openly; `D2` was resolved by ledger
replay with its losing branch recorded. **No findings.**

## §8 — the honest summary

**No. We are not closer to a curious humanoid that climbs the ladder than we
were yesterday, and today we are not closer to a longer list of green ticks
either — the list did not move at all.**

For a full day this project produced nothing: zero iterations, zero runs, zero
PASSes, one unchanged `HEAD`. The last time a *capability of Jack's* passed
anything was six days ago. Fourteen of the owner's twenty-four constitutional
commitments have a falsifiable claim registered and nothing behind it, and six
of those have a spec that is **runnable today** with no decision blocking it —
`smell`, `voice`, `balance`, `thermal`, `shelter`, `death & retry`. The
bottleneck is not science, not design, not the owner's decisions, and not
compute. **It is that the organ which does the work is switched off by a gate
that is metering the wrong thing, and the free GPU hours it would have spent
expire in 35 hours.**

The instruments remain genuinely good, and I want that on the record beside the
verdict rather than buried by it: the ledger re-verifies clean at 83 entries
with 81 controls probed, not one threshold has moved the wrong way in seven
days, all three constitutional ratchets are green, GPU accounting is exact, and
the `/usage` probes cost nothing. **This system's ability to tell the truth
about itself is in excellent condition. Its ability to *do* anything has been
zero for twenty-four hours.** That asymmetry — perfect instrumentation over a
stopped machine — is the whole finding, and `SYSTEM.md` already names it: *"when
the machine is sufficient, PROVE it by throughput."*

I will also note, as my predecessor did about itself, that **this audit spent
the meter it is reporting scarce.** That is the fourth consecutive one to do so.
`D13`'s change-gated no-op is armed and would have skipped some of them — but not
this one, and correctly so: the meters moved, and what they moved *to* is RANK 1.

---

# FOR THE BUILDER

Ordered by damage. **B1 and B2 execute during a pace skip and must** — they are
the only repairs that can run while the gate holds.

**B1 — rescue the orphan, then widen the rescue path (fourth audit asking).**

1. Commit `experiments/tests/sm_03_nose_reports_occluded.py` (32,086 bytes,
   untracked since 08-25 12:20). State in the commit message that its seed-90
   pilot **wrote zero bytes** — `/data/sm03_pilot_seed90.json.log` is 0 bytes and
   `/data/sm03_pilot_seed90.json` was never created — so the docstring's pilot
   numbers are **still owed** and its gates are **not frozen**. **Do not dispatch
   it until a pilot actually writes a result**, GPU clock notwithstanding.
2. In `ladder_loop.sh`, add a second, separately-committed rescue beside
   `harvest_bookkeeping()` for untracked/modified files under `experiments/tests/`
   and `docs/` — a mechanical *"orphaned work is versioned, never judged"* commit,
   reusing the torn-file guard and an explicit pathspec (the `add -A` ban stands).
   **Never** `experiments/ledger.json` by this path — only the runner writes
   verdicts.

**B2 — make the pace gate report its own recovery rate, and alarm when it is
losing (RANK 2, new).** This adds an observation, not a limit, and must run on
the skip path where nothing else runs. In `pace_gate`, keep the last reading in
a small state file and log, alongside the existing line:

```
PACING: … — recovery 0.387 pts/h, observed burn <X> pts/h over <N> h, gap <G> (WIDENING|closing)
```

When the observed burn has exceeded the recovery rate for **3 consecutive
readings**, escalate the wording to `LOCKOUT: the pace line cannot close this
gap` and append one line to `lost_iterations.log`, so a day of dark slots is a
**number** rather than a silence. This weakens nothing: it changes no branch, no
threshold, and no return value. It is the instrument whose absence let a
one-day lockout read as twenty-four routine skips.

**B3 — the loop-model pre-flight (RANK 1; carried from the 33rd audit's B6, and
its "the urgency was not [real]" caveat is now falsified — measured 87 → 99 in
six hours).** Before `run_claude`, read the loop model's *own* weekly line. If it
is ≥ 95%, log

```
ABORT: builder model <M> exhausted (<n>%) — the gate reads 'all models' (<m>%), which does not meter it
```

and exit 0 without consuming the slot. **It only ever refuses more than the 90%
stop, so it cannot weaken it.** Do this before touching the fallback chain: a
visible refusal is strictly better than a silent, undeclared switch to Opus.

**B4 — move the overseer off the Review's cron slot (carried, RANK 2 of the 33rd
audit, unrepaired).** `37 */6` and `37 6 * * *` collide at **06:37 daily**,
putting two concurrent long-effort Opus sessions on the shared meter 30 minutes
before the builder's 07:07 slot. Change to `37 3,9,15,21 * * *`: same cadence, no
collision, each audit mid-way between builder slots. Update `scripts/crontab.txt`
in the same commit — the live crontab and the file currently agree, and they must
keep agreeing.

**B5 — the change-gated overseer no-op (carried).** `D13`'s pre-registered
default, firing 2026-08-31: skip a slot when `HEAD` is unchanged **and** zero
builder iterations since the last audit **and** no `decide_by` before the next
slot **and** fewer than 3 consecutive skips. Conditions 3 and 4 are load-bearing.

**B6 — settle the three false `UNDECLARED` entries by document edit, not regex
(carried, and still correct).** Do **not** add `ANSWER` to `_SETTLED`:
`DECISIONS_NEEDED.md:1454` reads
`## D1 — DO NOT ANSWER "DO WHAT THE MEASUREMENTS SAY"`, and `_SETTLED` closes a
key if *any* header matches, so that widening silently closes **D1, the 38-spec
decision**. Append one settled header per entry using the token the tool already
owns (`RESOLVED`), quoting the ruling already in the body. Ratchet 3 → 0.

**B7 — `decisions.py` duplicate-key refusal (carried).** `_FIELD` matches any
indented `word: value` line as a **new key**, so a wrapped `default:` whose
continuation line happens to begin `default:`, `class:`, `blocks:` or
`decide_by:` silently *replaces* the field and the entry still reports as armed.
Refuse a duplicate key inside one `DECIDE:` block — raise, rather than
last-write-wins — and add a `_fixture()` case for it. **An armed default that
says the opposite of its author's intent will fire on 2026-08-31.**

**B8 — artifact check on every detached launch (third audit asking).** After
launching a detached pilot, wait 10–15 s and assert its log is **non-empty**
before reporting the launch succeeded. A 0-byte log is an import death and must
be reported as a failure **in the same iteration**. **RSS is not liveness** — the
`SM.03` pilot was 667 MB resident and wrote nothing.

**B9 — when the loop wakes, the first unit is `SM.03`'s pilot, not a new spec.**
A real seed-90 pilot, gates frozen from its numbers, then `dispatch.sh SM.03`.
It is the only registered `GPU_SHORT` claim spec standing between free hours and
a constitutional sense with zero passing claims. If the Kaggle window has already
closed by then, say so in the ledger rather than dispatching into an expired
quota.

---

# FOR THE OWNER

**Six hours ago my predecessor told you "no action is needed." I am withdrawing
that, and I am giving you the measurement it was missing rather than another
forecast.**

The 33rd audit was right to distrust the 32nd's extrapolation, and right about
the method — an audit should report a rate, not a date. But it then made a
forecast of its own ("the builder resumes unaided at ≈ Aug 27 05:07"), attached
the correct condition to it (*"it holds only while the foreign sessions stay
quiet"*), and **that condition failed within the hour.** Here are the three
facts, none of which is a projection:

1. **Your builder's own model quota is at 99%.** `week:Fable` — the model the
   builder runs on — read **99%** at 12:37. Nothing in the loop's control path
   reads that line; both gates meter `week:all models`, which reads 59% and says
   "hold budget for later in the week." **The budget being held is not in a
   currency the builder can spend.**

2. **The throttle has no way to catch up.** Its allowance rises at exactly
   **0.387 points/hour**. Measured burn over the last six hours: **1.17
   points/hour** — three times that — and the gap over the line widened from 8 to
   12. While the burn stays above 0.387, the gap *cannot* close and the builder
   *cannot* wake before the weekly reset. That is arithmetic plus a measurement,
   not a prediction, and it is checkable against the next six `PACING:` lines.

3. **The two clocks are 29 hours out of phase, and this needs no rate at all.**
   Your free Kaggle hours expire **2026-08-30 00:00 UTC**. Your Claude weekly
   meter resets **2026-08-31 04:59 UTC** — both timestamps published by the
   tools. **29.69 free GPU-hours die a day and five hours before the quota that
   would let anyone spend them comes back.** No sequence of events fixes that by
   waiting. This would be the fourth consecutive week; ~65 free GPU-hours
   cumulative.

**Where the spend is going, established rather than guessed.** No jackthelearner
organ runs on Fable except the builder — the overseer, the Review and the field
watch all run on Opus, which has no separate weekly line. The builder ran **zero**
iterations in the window. So **all 12 points of Fable burned in the last six
hours came from outside this project**; both long-running sessions in
`/home/opc` were active (last writes 12:09 and 11:53). Closing or pausing them
is the one lever that changes the arithmetic. I am telling you where the meter
went, not what to do with your own sessions.

**What I am NOT asking for.** I am not asking you to touch the 90% ceiling and I
would push back if it were proposed. That rule is yours and it is working
exactly as written.

**What is on your desk, newly armed as `D14`.** *When the builder's own model is
exhausted but the shared pool has headroom, what should happen?* Today the answer
is undeclared, and the loop silently falls back to Opus — the most expensive
model — on the very meter the gate exists to protect. My pre-registered default,
firing **2026-08-31** if you do not rule, is the **narrowing** one: the loop
refuses the slot loudly instead of switching models silently. It moves no
threshold and widens nothing. The option that would actually *save this week's
GPU hours* — running the builder unpaced for a bounded window, with your 90% stop
fully intact — is **yours to take and not available as a default**, because a
default may never widen what is permitted. It is written up in full in `D14`.

**Calendar note, unchanged:** six of your nine open decisions have
pre-registered defaults firing on **2026-08-31**, the same morning your meter
resets. All are reversible and none widens what is permitted — a crowded
calendar rather than a risk — but answering one or two before Sunday would thin
it out.
