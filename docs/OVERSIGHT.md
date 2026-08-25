# OVERSIGHT — 31st audit, 2026-08-25 18:50 UTC

## VERDICT: DRIFTING — the builder has not run for **six consecutive hourly slots**, the orphaned unit the last audit called "this iteration's unit; do not defer it" is still lying untracked in the working tree, and the gate that stopped the builder is metering spend the builder does not make

Four of the eight sections have **no findings**, and I put that first because a
clean result is worth as much as a dirty one when it is true:

- **§1 ledger integrity — clean.** `run verify` re-judged all **83** auditable
  PASS entries from the record alone and probed **81** controls: **0** verdicts
  that no longer re-derive, **0** gates that ignore their control, **0** controls
  declared but never run, **0** gates that could not be replayed, **0** entries
  that could not be audited, **0** controls run but undeclared. Two PASSes have
  no control at all (`T0.01`, `T0.10`) — both existence claims, both
  long-declared, unchanged.
- **§2 thresholds and controls — clean. Not one threshold moved in the loosening
  direction in seven days.** Examined by name in §2 below.
- **§7 bakeoff hygiene — clean.** Unchanged from the 30th audit; re-read, not
  assumed. One margin declared in the open (§7), not a violation.
- **The LC.03 data-starved question — I went looking for a violation and there
  isn't one.** Detail in RANK 3; it is the most important *negative* result in
  this audit and I state it plainly rather than dressing it as a concern.

The three mandatory ratchets are **all green**: `coverage` exit 0 (0 uncovered
commitments, 0 CLAIM-DEAD, 4 known-dangling GOAL citations, shrink-only);
`decisions --check` **"ratchet ok (3/10 undeclared)"** — down from 4, I armed one
(§6); `champions --check` "ratchet ok (6/8 seats with a phantom arena)".

**Nothing has been committed to this repository since 12:46 today.** The last
commit is the 30th audit's own report. That is the whole of RANK 1.

---

## RANK 1 — the repair for a stall was scheduled on the organ that was stalled, and no instrument noticed

The 30th audit (12:40) opened with: *"**B1 is this iteration's unit; do not defer
it.**"* It is now 18:50. Here is what happened in between:

| slot | outcome |
|---|---|
| 13:07 | `PACING … skipping` |
| 14:07 | `PACING … skipping` |
| 15:07 | `PACING … skipping` |
| 16:07 | `PACING … skipping` |
| 17:07 | `PACING … skipping` |
| 18:07 | `PACING … skipping` |

Six slots, zero iterations, zero commits. And the orphan is exactly where it was:

```
?? experiments/tests/sm_03_nose_reports_occluded.py   32,086 B   mtime 12:20
 M docs/LESSONS.md                                    +35 lines
   /data/sm03_pilot_seed90.json.log                   0 bytes    mtime 12:21
   /data/sm03_pilot_seed90.json                       still absent
```

**Six hours and fourteen minutes after the audit that ordered it rescued.**

**The structural point, which is the actual finding.** The builder is the only
organ that can execute a `FOR THE BUILDER` item. The overseer writes items; the
Review routes items; neither can commit a test file. So when the builder is
gated, the entire repair pathway of this system is gated with it — and the
gating decision is made by `pace_gate()` at 07 past the hour, *before* the
iteration ever reads `OVERSIGHT.md`. **The loop cannot distinguish "there is
nothing urgent to do" from "there is a flagged emergency and I am not allowed to
look at it."** The 27th audit's B3 fix anticipated half of this: a pace skip now
runs `harvest_bookkeeping()` so free bookkeeping is not blocked. But
`HARVEST_PATHS` is exactly three files —

```sh
HARVEST_PATHS="experiments/ledger.json experiments/gpu_budget.json experiments/gpu_submissions.jsonl"
```

— so the pace-skip path can rescue a detached ledger row and **cannot** rescue a
32 KB orphaned test file. The one path that runs while the builder is gated is
scoped to the one artefact that is not the problem.

**Why the gate fired, and the part that is genuinely wrong.** `pace_gate()` reads
`week:all models`, which is the **whole account**: the builder, this overseer (4
Opus runs/day at `37 */6`), the daily Review, and every other Claude session on
this box. It then throttles **only the builder**. There is no instrument anywhere
in the loop that separates builder spend from background spend.

The measurement that makes this concrete — five gated hours in which the builder
ran nothing at all:

| | 13:07 | 18:07 | Δ |
|---|---|---|---|
| `week:all models` (the gate) | 38% | 43% | **+5** |
| pace line `25 + ⌈65·elapsed/100⌉` | 38% | 40% | **+2** |
| gap (over the line) | 0 | **3** | **+3** |
| builder iterations | — | — | **0** |

The meter climbed two and a half times faster than the allowance it is measured
against, **with the builder completely idle**. The line rises 0.387 points/hour
by construction; the burn over that window was 1.0 points/hour. If that rate is
the new baseline the builder cannot get back under the line by not running, which
is the only lever it has. If it was a burst it will recover — and I cannot tell
you which, **because the instrument to tell them apart does not exist.** That is
the finding, not the extrapolation.

**I cannot decompose the meter and I will not pretend otherwise.** `claude_usage.py`
exposes exactly three lines — `session`, `week:Fable`, `week:all models`. There is
no per-organ attribution. The candidates for the background burn are this overseer
(4 Opus runs/day, ~10 min each at `--max-turns 60`, and the 12:37 run is inside
the window), the daily Review, and the two long-lived interactive `claude`
sessions running on this box under the owner's own account (pids 2926993, 3799119).
**I am one of the suspects and I cannot exonerate myself**, which is precisely why
the repair below is one I am asking for against my own organ.

**The visible consequence, which needs no extrapolation at all:**

| day | slots run | slots pace-skipped |
|---|---|---|
| 2026-08-24 | 16 | 3 |
| 2026-08-25 | **7** | **12** |

The skip rate went from 16% to 63% in one day.

---

## RANK 2 — six days without a claim PASS, and today the builder produced nothing at all

Unchanged at **84 PASS** — now for thirty-plus iterations, against a registry
that grew 169 → 187 in the same span (44.9% demonstrated, down from 49.1%).

- The last **claim**-kind PASS was `T3.01` (sight), **2026-08-20** — six days.
- The last PASS of any kind was `NE.00`, 2026-08-24 06:28, a `rule`-kind support
  spec that `coverage` itself lists as *"support passing, not credited"*.
- The nine most recent capability verdicts are **all red**: `T2.07` FAIL,
  `T3.07` FAIL, `LC.03 v2` VOID, `BA.02` VOID, `NE.01` FAIL ×2, `SH.01`
  ORACLE_CANNOT, `DP.05` FAIL, `T2.15` FAIL.

The 30th audit's RANK 2 said the ratchets move when a spec is *registered* and
never when a claim is *demonstrated*, and asked for **claim-kind PASSes per week**
as the counterweight (its B3). That is still the right instrument and it is still
unbuilt. But today sharpens it in a way yesterday's report could not: **today the
builder did not even register anything.** Yesterday's diagnosis was "the
scoreboard that moves is the wrong one"; today's is "no scoreboard moved,"
because RANK 1 stopped the organ that moves them.

`run status` totals, read live: `PASS 84, FAIL 9, VOID 3, NOT_RUN 0`, plus one
declared stale-by-content entry (`T2.02`, recorded VOID, flagged honestly by the
tool rather than hidden).

---

## RANK 3 — an owner-authored guard is prose, and D10's default would bypass it rather than overrule it

I opened this line of enquiry expecting a violation. **Most of it is not one, and
that result matters more than the residue.**

The owner flagged, on 2026-08-09, that a cheap bakeoff can crown the wrong
learning core, and three protections were written into `DECISIONS_NEEDED.md:483`.
Checking each against the live registry and the live ledger row:

1. **Scale-transfer check** (re-run the top two at ~10× on Kaggle, ranking must
   hold) — **alive and binding.** Carried verbatim inside `D10`'s armed default.
2. **The data-starved rule** (an arm failing the screen with a positive curve
   slope at cutoff is *not* eliminated) — **measured, disclosed, and bounded.**
   This is the one I expected to find quietly dropped. It was not.
   `{arm}/data_starved` is a real key on the LC.03 v2 row and it **fired on three
   of the four eliminated arms** — `ppo-needs` 1.0, `dreamer-xs` 1.0, `wm-efe` 1.0
   (`ppo-lp` 0.0, `wm-latent` 0.0, the sole 3σ learner). The fork that declined
   the re-screen was committed **2.5 days before the number landed**; `D10`'s body
   states all three flags in the open; and the refusal rests on a pre-registered
   general rule in `LESSONS.md` — *a screen with no re-screen cap is a ratchet*,
   because the 3σ bar retreats with added lives at the same speed the projected
   gain grows, so each re-screen is a fresh draw against a receding bar. **That is
   a cap decided before the miss, which is a measurement, not an excuse. Correct
   conduct, and I record it as such.**
   *One caveat, declared and not pressed:* the refusal prices the re-screen at
   *"~380 core-h, ~5 days of this 4-core box"* — the **CPU** option. The owner's
   clause specified **Kaggle**, where 29.7 free GPU-hours expire this Sunday. The
   σ-bar argument stands independently of cost, so this does not change the
   conclusion; but the cost half of the sentence answers a question nobody asked.
3. **The convergence check** (Addendum 2: declare a WINNER only if the runner-up's
   slope ≤ 0, **or** the projected crossover lies beyond 3× the tested budget;
   otherwise SPLIT-PENDING and extend both finalists) — **this one has no home.**
   I read `LC.04.notes` and `LC.05.notes` live today. LC.04 declares its arms and
   their parameter costs; LC.05 declares its four budgets and a ≤200-point
   decimated curve. **Neither carries the convergence rule.** No `_check` can
   enforce a rule that is not in the spec.

**Why that residue is time-critical rather than untidy.** `D10`'s default fires
**2026-08-31** and its branch (a) amends LC.04's premise to *"the screen IS the
arbitration when it returns exactly one"* — LC.04 never runs as a two-finalist
bakeoff. Addendum 2 binds the *winner* decision in LC.04/LC.05. If the default
fires with the rule still in prose, the convergence check is not overruled,
weighed, or set aside: it is **bypassed by construction**, because the experiment
it constrains is retired before it happens. A guard that is skipped rather than
failed leaves no trace in any instrument this system owns.

This is the **third instance this week** of the standing lesson *a prose-only
dependency is invisible to every graph ranking* (`a14d56d`) — after `DP.04`'s
LG.00 edge and T2.15's. It is the first where the invisible thing is an
**owner-authored guard** rather than a dependency edge, which is why it ranks
here rather than as a footnote.

**Armed as `D12` this audit** (see §6). Its default only moves the owner's own
rules from prose into the place gates bind. Nothing is weakened.

---

## RANK 4 — the builder's own model meter is at 72% and the loop prints it hourly under the words "not the gate"

| meter | reading | week elapsed |
|---|---|---|
| `week:all models` (**the gate**) | **43%** | 22% |
| `week:Fable` (**what the builder actually spends**) | **72%** | 22% |

The loop runs `JACK_LOOP_MODEL=fable`. `usage_gate()` and `pace_gate()` both read
`week:all models`. That is *correct as designed* and it is what the standing note
says to act on — but it means the 90% hard stop cannot see the meter that will
actually run out first. Trajectory, from the loop's own PACING lines:
**54% (08-24 22:11) → 72% (08-25 18:07)** = +18 points in 20 hours, ≈0.9/hour. At
that rate Fable reaches 100% around **2026-08-27**, with four days of the week
left.

**What happens then is not a stop — it is a silent substitution.** The gate reads
43% and says proceed; `lib_credits.sh` walks the fallback chain to another model;
and the builder's model changes mid-week with nothing in the ledger to say so.
This is not hypothetical: **2026-08-24 12:08, 13:07 and 14:07 all ended `rc=1`**
after exhausting every model in the chain. That machinery worked — it logged to
`lost_iterations.log` and the 15:07 iteration inherited them — but it fired
*after* exhaustion, not before.

The PACING line dutifully prints `week:Fable 72% (not the gate)` every hour. It
is honest and it is inert: nothing anywhere consumes that number.

---

## RANK 5 — W34 GPU: 0.31 h of 30 spent, five days to expiry, and the only candidate is the orphan from RANK 1

| week | charged | of 30 |
|---|---|---|
| 2026-W32 | 16.61 h (17 jobs) | 55% |
| 2026-W33 | 7.89 h (22 jobs) | 26% — **22.1 h expired unused** |
| 2026-W34 | **0.3111 h (1 job)** | **1%** |

The single W34 charge is `jack-ladder-1787631708` (T2.15, `ok: true`) and it
produced a real FAIL row with every gate green — **spend accounted for, receipts
committed, no waste.** `gpu_submissions.jsonl` joins cleanly; every `attempt` has
a `result`.

`SM.03` is the declared `GPU_SHORT` candidate and it cannot dispatch: its
implementation is untracked, `dispatch.sh` refuses an unpushed HEAD, and its pilot
is dead. **The blocker on this week's GPU quota is RANK 1**, not a shortage of
work and not dishonesty — the builder declined to manufacture a dispatch four
times this week and said so in writing each time, which is the opposite of gaming.

---

## §2 — thresholds and controls, seven days, examined by name

No commits have touched `experiments/` since the 30th audit, so this window is
that audit's plus nothing. I re-ran the diff independently rather than inheriting
the conclusion. Every numeric constant that changed is either a **new spec's
pre-registration** (T2.15, DP.05, NE.00, NE.01, SH.02, SM.03, LG.*) or a
**tightening**:

- `t0_21_coverage_audit_honest.py`: `N_PROPERTIES` **9 → 10 → 11**. The guard
  gained properties; each re-ran PASS and re-stamped.
- `registry_expansion.py` `ed2d969`: `DP.04.depends_on` gained `LG.00` — a
  **blocking edge added**, discharging an instruction standing in that spec's
  notes since 2026-08-10.
- `ne_01_nobody_survives_by_accident.py`: `DELTA_T_NIGHT` **12 → 10**, a
  declared pre-run calibration with its sweep table shipped in `metrics`, the
  0.3–0.6 gate untouched — and NE.01 FAILed anyway, twice, on identical digits.

**No `_check` gained an `or`. No control was deleted or weakened. No seed count
was reduced. No assertion was removed.**

**One small guard-of-the-guard note, not a violation and not urgent.** `b624d78`
generalised T0.21's P6 so its removal set is *computed* rather than a cached pair
— the right repair, and it caught a real staleness. But the set is now computed by
calling `cov()`, the function P6 exists to test:

```python
minus = {k: v for k, v in fix.items() if k not in set(cov(fix, "shelter/building"))}
if cov(minus, "shelter/building"): failed.append("p6_deleted_spec_loses_coverage")
```

If `cov` ever returned empty, `minus == fix`, the assertion is falsy, and **P6
passes vacuously** — the one shape it exists to catch. P1–P5 would catch a
totally-dead `cov`, so the hole is narrow and I am not claiming it is live. Worth
one line asserting the computed removal set is non-empty before using it.

## §3 — drift from the goal

**No drift, because the builder did no work today to drift with.** The last unit
(12:07, SM.03) traces cleanly to GOAL.md:45–48 — *"olfaction finds food, fire and
decay at a distance and through occlusion — the sense that works when sight
fails"* — and its design (supervised readout around SM.02's measured RL
bottleneck, held-out layouts, occluder-removed alive-proof) is sound. I read it.

**The converse, which is the harder question.** From `coverage`: **14 of 23
commitments have live claim specs and nothing passing.** The three GOAL.md claims
most at risk of quiet neglect, unchanged from yesterday and one day older:

- **Curiosity** — 12 specs, **1 pass** (`T2.08`, a coverage metric). `T5.06`
  *"unprompted exploration is real"* is `not implemented`. The ladder-and-apple
  sentence itself has no passing spec.
- **One brain / all senses in unison** — 21 specs, **1 pass**, and that one
  (`UB.9`) is registry-declared *conditional* pending a per-arm descent re-run.
- **Learning by living** — `death & retry` 3 specs 0 pass; `hunger/thirst` 5
  specs 0 pass; `shelter/building` 1 spec 0 pass; `fast/slow` 8 specs 0 pass.
  The four commitments closest to *"he gets thrown in, figures life out or
  doesn't, dies, and tries again"* hold **zero** passing claims between them.

Newly registered and all `not implemented`: `LG.00`, `LG.01`, `LG.02`, `LG.10`,
`SH.02`, `SM.03`. The denominator grew; nothing behind it has run.

## §4 — is the builder alive and productive?

**Alive, disciplined, and stopped.** In the 24 h to 18:50: **25 slots — 10 ran,
15 pace-skipped**, all 10 ending `rc=0`. Zero `rc=1`. No repeated identical
failures, no paused loop, no iterations aborting on load (max 0.16), no credit
exhaustion in the window. **PASS delta: 0.** The six most recent slots are the
unbroken skip run in RANK 1. The cron entry is present and firing (`7 * * * *`);
this is a gate decision, not a dead loop.

## §5 — compute honesty

RANK 5. Spend is fully accounted; the waste is unspent quota, not unexplained
hours.

## §6 — stuck decisions

**Zero `MEANS-ESCALATED`** — the D1 disease stays cured. **Nothing overdue.** No
owner decision was quietly acted on without being recorded.

**Seven armed decisions now default on 2026-08-31**, six days out: `D1` (costs
**38 specs**), `D4` (8), `D10` (8), `D7`, `D8`, `D9`, `D11`. The 30th audit's
RANK 4 finding stands and I re-verified it: **D1's default names a bakeoff over
four arms (A-prime, B, C, D) that does not exist as a spec, a test file, or a
queue row.** If 08-31 arrives unanswered it resolves the constitutional half
correctly and leaves 38 specs blocked with the deadline spent.

**I armed one, per the standing duty — `D12`** (RANK 3), and the arming is real
this time: `decisions --check` went **4 → 3 undeclared**, re-verified live. Class
`goal` — it asks what standard of evidence an adopted learning core must clear,
which no experiment answers. Deliberately **not** `means`: LC.04 is blocked behind
LC.03's VOID, there is no bakeoff to run, and classing it `means` would raise a
blocking MEANS-ESCALATED violation against an entry with nothing to escalate. Its
default only transcribes rules the owner already wrote into the place gates bind;
it weakens nothing and moves no threshold.

**The renumber, disclosed.** Arming required giving the heading a D-number,
because the parser bug the 30th audit found is still open: `parse()` keys an
un-numbered heading by a 52-character slice of its title (spaces included) while
`_DECIDE`'s id class is `[A-Za-z0-9._-]+` (spaces excluded), so a title-keyed
entry can never carry a `DECIDE:` block. **Two of the three remaining UNDECLARED
entries are in that state and cannot be armed by anyone.** Both are also already
answered in their own bodies (*"The owner's hands"* — DECIDED YES 2026-08-09;
*"Was physics-first retired"* — DECIDED (a) RUN IT 2026-08-09). That fix is
builder item **B2** and it is now on its second carry.

## §7 — bakeoff hygiene

Re-read, not inherited. Three entries, all sound. `PS.01/J` recorded **VOID** for
arms below the 3.0σ learning gate and was correctly **not** treated as a verdict —
it was re-run as `J2`. `D2` was resolved by ledger replay with the method, the
losing arm, and a **re-open trigger** all recorded. One margin stated in the open:
`PS.01/J2`'s winner `impact_speed` beats runner-up `peak_dvel` by **2.66σ**, under
this project's own 3.0σ ruler — but the declared gate is over the *null* (10.32σ)
in `screen` mode, the arms are deterministic reductions of identical cached
rollouts with no training that could have failed, the file says so in its screen
rationale, and the runner-up margin is *reported*, not gated. Declared, not
hidden. Not a violation.

---

## §8 — THE HONEST SUMMARY

**No.** Not closer to a curious humanoid, and today not even closer to a longer
list of green ticks. The list has been frozen at **84** for thirty-plus
iterations, and today it did not move because **the builder ran for zero of the
last six hours.**

Yesterday's audit said the machine had become excellent at telling the truth
while the creature stopped moving. Today that is still true and it has acquired a
sharper edge: **the machine's excellence is now what is stopping the creature.**
Every organ behaved correctly this afternoon. `pace_gate()` read its meter, named
it, printed both lines, and skipped — exactly as designed, and the design is a
good one that fixed a real problem. `harvest_bookkeeping()` ran on the skip path,
exactly as the 27th audit ordered. The overseer wrote a precise, correct,
well-evidenced repair item and marked it *do not defer*. And the net effect of
all that correctness was six hours of nothing, with a 32 KB unit of real work
lying untracked on the floor the whole time, in a week when 29.7 free GPU-hours
are five days from expiring.

That is a new failure shape and it deserves its name. This system has learned,
over thirty audits, to make every claim it makes checkable. It has not learned to
notice **an absence** — a launch that produced nothing (30th audit, RANK 1), a
guard that was skipped rather than failed (RANK 3 today), an hour in which the
organ that does the work was not permitted to look at its own emergency queue
(RANK 1 today). Every instrument here reasons about rows that exist. The same
blind spot that let four commitments sit uncovered on 2026-08-10 is still here,
wearing its third set of clothes this week.

Apply the ladder-and-apple standard, which is the only honest ruler: nothing in
the ledger shows Jack trying to climb anything out of curiosity, falling, and
trying again. `T5.06` — *unprompted exploration is real* — is `not implemented`.
Twelve curiosity specs, one pass, and that pass is a coverage metric. The five
instruments that all point the same way — `SH.01` ORACLE_CANNOT, `LC.03 v2`'s
single learner, `DP.05`'s planners-eat-reactives-never, `NE.01`'s freeze-or-cook
shelter, `T2.15`'s router losing to bag-of-words — still say the world is too
shallow and the core cannot climb what gradient it has. That is `D10`. It is
armed. It defaults in six days, and this afternoon nobody was allowed to work
on it.

---

## FOR THE BUILDER

Ranked. **B1 and B2 are one unit and they are this iteration's work.**

**B1 — Rescue the orphan, then make the gate unable to block its own repair.**
Four parts, in order:

  (a) **Commit the orphaned 12:07 unit.** `experiments/tests/sm_03_nose_reports_occluded.py`
      (untracked, 32,086 B) and `docs/LESSONS.md` (+35 lines). **Semantically
      diff before you act** — this is your own timed-out unit, not damage — then
      commit both **by explicit pathspec**. Do not `git add -A`. This has now
      been owed for over six hours.

  (b) **Add an emergency lane to the pace-skip path.** This is the durable half
      and the point of the item. `pace_gate()` decides at 07-past *before* the
      iteration reads `OVERSIGHT.md`, so a flagged emergency cannot be seen, let
      alone acted on. Cheapest honest form: on the skip path, after
      `harvest_bookkeeping()`, check whether the working tree is dirty outside
      `HARVEST_PATHS`; if it is, **log it loudly** (`PACE-SKIP: tree dirty outside
      the harvest paths — N file(s), oldest mtime T`) and do not skip silently.
      An orphan that survives one skip is an accident; one that survives six is a
      missing instrument. Do **not** auto-commit unknown files from the unattended
      path — the `add -A` ban (`c0afded`) exists for exactly that, and it swept in
      both directions. Report, don't sweep.

  (c) **Re-launch the SM.03 pilot through `scripts/launch_detached.sh`** — the
      tool that exists for this and was not called. Confirm the `LAUNCH …` header
      line is in the log and the pid is alive at 15 s **before** reporting
      anything. `/data/sm03_pilot_seed90.json.log` is still 0 bytes; treat the
      12:07 launch as never having happened. Then it can dispatch to Kaggle
      against W34's 29.7 expiring hours.

  (d) **Still owed from the 30th audit's B1(c):** a `LAUNCHED:` receipt line
      (spec id, log path, pid, ISO timestamp) written by `launch_detached.sh`,
      plus a startup check that reports any receipt older than N hours whose log
      is still empty or whose pid is dead with no ledger row. Nothing in this
      system watches for the absence of a result.

**B2 — Separate builder spend from background spend, and fix the parser bug.**
Two parts, both cheap, both blocking-in-effect:

  (a) **Instrument the meter (new, and it is the RANK 1 repair).** `pace_gate()`
      throttles the builder on `week:all models`, which includes this overseer
      (4 Opus runs/day), the daily Review, and every other Claude session on this
      box. Nothing separates them. Log the reading of **both** meters at iteration
      **start and end** to a receipts file (one JSON line: iso, slot, phase,
      all_models_pct, fable_pct, week_elapsed). Two days of that turns "the meter
      went up while the builder was idle" from an inference into a measurement,
      and tells you whether the afternoon of 2026-08-25 was a burst or a baseline.
      **I am asking for this against my own organ** — if the overseer's four daily
      Opus runs are eating the builder's pace allowance, that is my footprint and
      it should be measurable, not argued.

  (b) **Fix `decisions.py`'s unarmable-entry bug** — second carry. `parse()` keys
      an un-numbered heading by a 52-char title slice (spaces included) while
      `_DECIDE`'s id class is `[A-Za-z0-9._-]+` (spaces excluded), so a title-keyed
      entry can never be joined to a `DECIDE:` block. Two of the three remaining
      UNDECLARED entries are in that state. Either emit a distinct violation
      (`UNARMABLE: give this heading a D-number, it cannot carry a DECIDE block`)
      or slugify the title key and accept the slug in `_DECIDE`. **The tool must
      be unable to report an unarmable entry as merely unarmed.** Add it as a
      known-answer case. Then move the two answered entries out (*"The owner's
      hands"* — DECIDED YES 2026-08-09, tracked at `INTEGRATION_QUEUE.md:445`;
      *"Was physics-first retired"* — DECIDED (a) RUN IT, and record that it is
      **blocked in fact** behind `T2.01` FAIL via `T5.01.depends_on`, so the
      ruling is honoured by the graph) and the superseded `D3 (original)` heading,
      then **re-baseline `BASELINE_UNDECLARED`** on the true count or the shrink
      reads as slack that was never won.

**B3 — Transcribe D12's guards before 08-31.** The convergence check and the
data-starved rule live only in `DECISIONS_NEEDED.md` prose; `LC.04.notes` and
`LC.05.notes` carry neither. Write them in. If `D10`'s default fires first, LC.04
never runs as a two-finalist bakeoff and the convergence check must land on the
CHAMPIONS learning-core seat instead, as a pre-condition of adoption. This is
transcription, not design — the owner already wrote the rules.

**B4 — Measure claim-kind PASSes per week** (30th audit B3, second carry). Every
ratchet moves when a spec is *registered*; none moves when a claim is
*demonstrated*, so a week of pure registration reads identically to a week of
progress. One line of arithmetic over `ledger.json` × `COVERS: … (claim)`.

**B5 — Write D1's bakeoff, or say in `DECISIONS_NEEDED.md` what firing its
default without one actually buys.** Six days; 38 specs downstream; the four arms
it names exist nowhere.

**B6 — UB.9's per-arm descent measurement.** Sixth carry. Its PASS is
registry-declared *conditional* and it is one of only two passing claims under
*one brain / unison*.

**B7 — One line in T0.21's P6**: assert the computed removal set is non-empty
before using it (§2). Small, not urgent, and it closes the vacuous-pass shape in
the guard that exists to catch stale credit.

## FOR THE OWNER

Two things, and one of them may be you.

1. **Your builder ran for zero of the last six hours, and the gate that stopped
   it counts spend from every Claude session on this box — including your own
   interactive ones.** `pace_gate()` meters `week:all models` (43%) and throttles
   only the loop. Between 13:07 and 18:07 that meter rose **5 points while the
   builder ran nothing at all**; its allowance rose 2. I cannot tell you who spent
   it — the CLI exposes no per-organ attribution, and the overseer's own four
   daily Opus runs are among the suspects. **If you are running long Claude
   sessions on this box, they are silently costing the builder its hours.** Two
   levers exist today and both are yours: `.usage-resumed` (with a ceiling and an
   expiry) suspends pacing, and `JACK_PACE_FLOOR` raises the opening allowance.
   I have not touched either. B2(a) asks the builder to instrument this so the
   next audit can answer the question instead of raising it.

2. **Seven decisions now default on 2026-08-31** — `D1` (38 specs), `D4` (8),
   `D10` (8), `D7`, `D8`, `D9`, `D11`, and I armed an eighth (`D12`) onto the same
   date. That is one sitting and it is the sitting that unblocks the frontier.
   Read `D10` first: five independent instruments now measure the same thing —
   the world is too shallow to reward the behaviours we are asking Jack to learn.
   Two warnings about letting the clock do the work for you:
   - **`D1`'s default fires into an unwritten experiment.** It correctly strikes
     the unconstitutional option and then hands the rest to a four-arm bakeoff
     that does not exist. You would buy a ruling and no motion; 38 specs stay
     blocked with the deadline spent.
   - **`D10`'s default would bypass one of your own guards.** You wrote, on
     2026-08-09, that no LC winner may be adopted while the runner-up's learning
     curve is still climbing toward it (the convergence check). `D10` branch (a)
     retires the two-finalist bakeoff that guard was written to constrain, so
     firing it doesn't overrule the guard — it removes the thing the guard
     attaches to. I armed `D12` to transcribe your three guards into the registry
     so this cannot happen by silence. **One line from you drops the convergence
     check instead, if that is what you want** — but then the record shows a guard
     retired by a ruling rather than by a deadline passing over prose.

3. **The honest state, in one sentence:** the measurement apparatus is in
   excellent health, nine consecutive capability verdicts have come back red and
   informative, and today the apparatus itself is what kept the builder from
   working.

---

### A note on what I did not commit

`docs/LESSONS.md` still carries **uncommitted builder work** (the `[s]`-tier entry
from the 12:07 iteration), and `experiments/tests/sm_03_nose_reports_occluded.py`
is still untracked. Committing either would sweep in-flight builder work into an
overseer commit — the failure `c0afded` banned `git add -A` for, in both
directions. I have left both exactly where the builder left them, for the second
audit running, and escalated the fact that nothing in the loop will pick them up
to **B1(b)**. This commit stages **only** `docs/OVERSIGHT.md` and
`docs/DECISIONS_NEEDED.md`, by explicit pathspec.
