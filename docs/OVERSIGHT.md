# OVERSIGHT.md — independent audit of the JackTheLearner system

> Current-state report, rewritten each run by `scripts/overseer.sh`. Not a log.
> The overseer reads and reports; it does not implement, re-run, or fix science.

**Date:** 2026-08-11 17:05 UTC (6th audit; previous 2026-08-10 12:45 at `90934f1`)
**HEAD:** `7e6fc60`, tree clean, `origin/main..HEAD` empty · ladder **66 PASS /
162 registered** (1 FAIL `T2.01`, 1 VOID `T2.02`, 1 ERROR `T1.02`) · Kaggle
**18.04 of 30 h remaining, expiring 2026-08-16** · usage override expires
**2026-08-12T12:00 UTC — 19 hours from now**

## VERDICT: DRIFTING

**The ledger itself is in the best shape it has ever been, and I can prove it
with commands rather than assertions.** `run verify` re-judged **65 PASS entries
from the record alone** through their committed `_check` functions: 0 verdicts
that no longer re-derive, 0 gates that ignore their control, 0 controls declared
but never run, 0 unreplayable, 0 unauditable. All 69 recorded commits still
resolve in git. **Zero `+dirty` stamps on the board** — yesterday's live defect,
closed. `experiments.coverage` exits 0: no commitment has zero specs. Section 2
is clean for the **sixth** audit running. Four of the ten items I left for the
builder yesterday closed, in roughly **one hour** of builder runtime.

The drift is not in the science. It is in three places:

1. **`coverage.py`'s pass-count credits world-fidelity and harness certificates
   as if they were capability — including for `one brain / unison`, the
   constitution's headline.** The builder fixed exactly this class inside
   `senses.py` yesterday (the `SENSOR` vs `LOAD-BEARING` tier, because `PG.6`
   was marking sight `[PASS]`). The instrument this charter orders me to run
   FIRST did not get the same tier, so the two organs now disagree in print:
   `run senses` says **0/10 LOAD-BEARING**; `coverage` says sight has 2 passes
   and unison has 1. (§1a, **RANK 1**)

2. **`run blocked`/`run next` decide a dependency is satisfied by
   `status == PASS` — the precise rule that `borrow_metrics` and T0.22 were
   built to declare insufficient, one function away in the same file.**
   Consequence today: `PS.01` and `XL.00` are both stale, `XL.00` *depends_on*
   `PS.01` and now *borrows* from it, so the runner will offer `XL.00` as
   runnable and the guard will VOID it. The ladder's headline result — death,
   respawn and the diary crossing it — plus the entire `LC.03`/`LC.04` bakeoff
   branch sit behind that. (§1b, **RANK 2**)

3. **66 green ticks and tiers 3, 4 and 5 are at exactly zero.** PASS by tier:
   **T0 = 23, T1 = 12, T2 = 30, T6 = 1; Tier 3 (earn your parameters) = 0,
   Tier 4 (unison) = 0, Tier 5 (the claims) = 0.** Every capability the project
   exists to prove lives above a line the ladder has not crossed. (§8)

And one thing only the owner can fix: **the loop was dead for 22 h 53 m of the
last 24 hours**, restarted by hand, and the grant that restarted it **expires in
19 hours** (§4, FOR THE OWNER 1).

---

## 1. Integrity of the ledger — CLEAN

**69 entries, 66 PASS.** Checked mechanically:

| check | result |
|---|---|
| PASS entries with no implementation in `experiments/tests/` | **0** / 66 |
| recorded `commit` that no longer resolves in git | **0** / 69 |
| entries carrying a `+dirty` stamp (yesterday: 3 in flight) | **0** / 69 |
| `run verify`: verdicts that no longer re-derive from the record | **0** / 65 |
| `run verify`: gates that IGNORE their control | **0** (63 probed) |
| `run verify`: controls declared in the spec but never run | **0** |
| `run verify`: controls run but NOT declared | **0 / 0** |
| PASS entries with no control at all | **2** — `T0.01`, `T0.10` |
| ledger ids not in the registry / registry ids double-counted | **0 / 0** |

`T0.01` (13 modules import) and `T0.10` (Kaggle job round-trip) remain the only
two claims whose gate has never been shown capable of reporting the bad case.
`T0.01` I accept — a failed import is self-evident. **`T0.10` I do not, sixth
audit:** "a job round-trips" is exactly the shape a broken detector reports as
healthy, and this project has already been bitten by it (`DECISIONS_NEEDED.md`
line 14: a Kaggle kernel that ran to `COMPLETE` with **no GPU attached** and
nothing signalled it).

**Coverage: `experiments.coverage` exits 0.** No commitment has zero declared
specs. **12 of 23 commitments have specs but nothing passing** (down from 13):
touch/contact, balance, damage/nociception, shelter/building, tool use, voice,
proprioception, thermal, sleep, social/other agents, plasticity, generality.

### 1a. RANK 1 — the gap-finder counts certificates as capability

Yesterday I asked for `COVERS:` to be made load-bearing so a regex could not
invent coverage. **That landed and it works** — `honest`→`nest` is dead, and
`coverage.py` now prints nominations in a separate column with the correct
sentence beneath it (*"A nomination is NOT coverage"*). The **false-positive
half moved one level up** and is still live: `n_pass` is a flat count of PASS
entries, and a PASS can mean four very different things.

I inspected **every spec `coverage.py` currently credits with a PASS**:

| commitment | reported | the passing spec | what it actually certifies |
|---|---|---|---|
| **one brain / unison** | 21 specs, **1 pass** | `LC.01` | *"Every candidate core takes every sense into one latent, or it is not a candidate"* — an **admission rule applied to bakeoff arms**. No core has been adopted. It certifies the screening rule, not a fused brain. |
| **curiosity** | 12 specs, **1 pass** | `PG.4` | *"Noisy-TV panel traps naive curiosity"* — a property of the **world**. Carried from audit 5, unchanged. |
| **fast/slow** | 5 specs, **1 pass** | `DP.00` | *"**This world** rewards looking ahead at all"* — its own hypothesis says the finding is about the world, not about Jack. |
| **sight** | 5 specs, **2 pass** | `PG.6`, `PG.9` | the **playground's camera** resolves radius/bearing, and is not pointed into a wall. |
| **hearing** | 6 specs, **2 pass** | `PG.5`, `PG.7` | the **playground's audio** pans correctly, and a fixture leaks only its intended bit. |
| **language (parent)** | 5 specs, **1 pass** | `T1.13` | *"The grounding pairs are real"* — a **training-data honesty** check. |
| **smell / taste** | 2 & 3 specs, 1 pass each | `SM.01`, `TA.01` | **sensor and fixture** certificates. `run senses` correctly calls both `SENSOR`, not `LOAD-BEARING`. |
| **hunger/thirst** | 2 specs, 1 pass | `PS.01` | a **drive calibration** (`j0`, `alpha`) consumed by other specs. |
| **memory across lives** | 3 specs, 2 pass | `ME.10`, `XL.00` | **genuine Jack-level claims.** |
| **death & retry** | 2 specs, 1 pass | `XL.00` | **genuine.** |

So of the twelve commitments coverage reports as having something passing,
**three are carried by real capability claims and nine are carried by the
world, the harness, the training data, or a sensor.** The reader of that table
cannot tell which is which, and the table is the first thing this charter says
to look at.

**Why this is RANK 1 and not pedantry.** The builder proved yesterday that it
takes this distinction seriously: `senses.py` grew a fourth tier *precisely*
because `DEMONSTRATED` meant "some declared spec is PASS" and `PG.6` was
therefore marking sight green. `T0.20` gained P7 and is checked in **both**
directions so the top tier cannot be unreachable by construction. That is the
right fix, done well — and it was applied to the ten senses only. The identical
defect, in the identical shape, survives in the instrument that maps **all
twenty-three** commitments, and it is now flattering the one line SYSTEM.md says
no bakeoff may trade away: **`one brain / unison` reads "1 passing" on the
strength of an admission rule for candidates that have not been chosen.**

The two organs already contradict each other in print. `run senses` ends with
*"0/10 are LOAD-BEARING — the standard GOAL.md sets: none."* `coverage` ends
with a table showing sight and hearing at 2 passes each. Both are reading the
same ledger.

### 1b. RANK 2 — the dependency graph still satisfies on `status == PASS`

`experiments/run.py:590`, inside the blocker walk:

```python
for d in spec.depends_on:
    if ledger.status(d) is Status.PASS:
        continue
```

That is the rule T0.22 was registered to retire. `LESSONS.md` states the
replacement in its own words: *"a value borrowed from another spec's ledger
entry needs the same freshness check the scoreboard applies to the entry
itself"*, and the fix — `protocol.borrow_metrics`, refusing on NOT PASS, DIRTY,
UNVERIFIABLE, CHANGED, missing and non-numeric — is real, guarded, and correct.
It was installed on the **borrow** path. The **dependency** path, forty lines
away in the same file, still asks the old question.

**This is not hypothetical; it is today's board.** `run stale` names two:

```
PS.01  PASS  ps_01_drive_calibration.py: ran on 2cd686a3…, now 94735681…
XL.00  PASS  xl_00_death_and_respawn.py: ran on 5716b043…, now 811833ea…
```

- **`PS.01`'s flag is definitional and two audits old.** Cause traced: `74f8631`
  added `IMPL_DEPS = ["playground.py"]` to the file after its run at `248b160`;
  `git diff 248b160..HEAD` on that test is *only* the five-line declaration, and
  `playground.py`'s last modification is `29d189f` (08-10 08:06), which predates
  PS.01's run (08:32:26). No world change is hiding behind it. It was **FOR THE
  BUILDER §3 yesterday** and is the one of that trio that did not get re-run.
- **`XL.00`'s flag is real.** Its implementation changed at `83fa1a5` to route
  `_calibration()` through `borrow_metrics` — the RANK 2 repair I asked for,
  done thoroughly, provenance recorded into XL.00's own metrics.

Compose the two and you get the failure the runner cannot see:
`XL.00.depends_on == ["LC.02", "PS.01"]`, PS.01 is PASS, so **`run next` and
`run blocked` will present XL.00 as runnable** — and the moment it runs,
`borrow_metrics` refuses the stale source and `_check` returns **`Status.VOID`**
on `calibrated != 1.0`. The builder would burn an iteration to learn an ordering
that was already computable: **PS.01 must be re-run before XL.00.**

The blast radius is larger than one spec. `LC.03.depends_on ==
["LC.00","LC.01","LC.02","PS.01","XL.00"]` and `LC.04.depends_on == ["LC.03"]`,
and both score `life_gain` in the world PS.01 calibrates. **The project's
biggest open decision — the learning-core bakeoff — sits downstream of two stale
entries that the dependency graph reports as satisfied.**

Generalised: `depends_on` is an edge between **specs**; a stale entry is a fact
about a **ledger row**. The system now has one rule for "is this number usable"
(`staleness_of`, called by both `stale_claims` and `borrow_metrics` — correctly
unified) and a *second, older* rule for "is this dependency met". Two functions
answering "the same" question is the defect `LESSONS.md` already priced once, at
twelve specs flagged stale in perpetuity.

---

## 2. Thresholds and controls over time — CLEAN (sixth audit running)

Window audited in full: `90934f1..HEAD` (everything since audit 5) — 9 files,
**+2,221 / −92 lines**. Prior windows were audited across audits 1–5.

**Every deleted line in the registry diff is reformatting.** I read all 92:
each `-` is a `Spec(...)` closing line reopened to append `notes="COVERS: …"`.
The two that look alarming out of context —
`- control="Shuffled (command, action) pairing must collapse to chance."` and
`- control="Sleeping with the rehearsal buffer EMPTIED must forget."` — are both
immediately re-added with a trailing comma on the next line. **No control
deleted. No `seeds=` reduced** (`seeds=3` appears on both sides of every hunk it
touches). **No assertion removed. No `_check` gained a permissive `or`.**

**One threshold moved, in the tightening direction, and it is stated in its own
commit message.** `T0.20` `falsified_by` went **6 → 7 properties** (`7e6fc60`).
The new P7 is the load-bearing tier check and it is written **in both
directions**: a stub ledger where only `SM.01` passes must read `SENSOR` for
smell, and the *same* ledger with `SM.02` passing must read `LOAD-BEARING`.
That second half is what stops the new top tier from being unreachable by
construction — a green light spelled differently. This is the discipline
`LESSONS.md` prescribes, applied without being asked.

**New tests in the window are additive and self-adversarial.** `T0.22` scans
every ladder test for a direct `results.get("<spec id>")` lookup and requires
zero, so the guard closes the class rather than the instance; `T0.21` keeps the
failed regex rule executable as its own control; `SM.01` replaced a
re-derivation of its own falloff formula with a **named rival model**
(inverse-square, must miss by ≥0.10, measured 2.57) after finding the first
version scored 0.0 against itself.

**No threshold has ever been moved in the loosening direction, no control
weakened, no assertion removed, across the project's entire life and six
audits.** That remains the single most valuable fact in this report.

One thing I looked at hard and cleared: `bakeoff.py`'s **`screen` gate mode**,
which bypasses the 3σ learning gate, is *not* a loophole. It requires a
non-empty `Spec.screen_rationale` (`ValueError` otherwise), requires a minimum
number of arms to clear before it may crown anything, does not change the null,
and is itself guarded by `T0.19` (PASS). The one use on the board records its
rationale in `DECISIONS_RESOLVED.md`.

---

## 3. Drift from the goal

**Builder work since audit 5, each traced to a GOAL.md sentence — zero drift:**

| work | GOAL.md sentence |
|---|---|
| `DP.00` PASS (lookahead pays here, +75.8 steps at 4.31σ) | *"Fast and slow, in one brain"* (owner, 2026-08-10) |
| `TA.01` PASS (poison fixture, visually identical twin at 0.51) | *"gustation drives conditioned taste aversion… the fastest learning in biology"* |
| `SM.01` PASS (odour field; hidden source at 477× noise floor) | *"olfaction finds food, fire and decay… the sense that works when sight fails"* |
| `T0.22` PASS + `borrow_metrics` | *"protects the honesty of watching what happens"* |
| `T0.21` PASS, `COVERS:` made load-bearing | the coverage instrument — audit 5 RANK 1 |
| `senses.py` fourth tier, `T0.20` P7 | *"ablate a sense, something measurable must degrade"* |
| `lib_usage.sh` — the 90% stop's resume | keeping the machine alive under the owner's own rule |

**The converse — GOAL.md with nothing passing behind it.** 12 of 23
commitments, and the ones that matter most are unchanged from yesterday:

- **`one brain / unison` — 21 specs, and the "1 pass" is `LC.01`, an admission
  rule (§1a).** On the honest reading this is still **0**. `UB.14` — `CPU_LONG`,
  depends only on `PG.1` — has now been runnable since 2026-08-09 14:22, **51
  hours**, and remains untaken.
- **`curiosity` — 12 specs, and the "1 pass" is `PG.4`, a world certificate.**
  Nothing on the board shows Jack **choosing** to do anything. Second audit
  saying this.
- **`plasticity` — 0 of 2**, despite PLASTIC-ONLY being a constitutional decree.
- **`generality` — 0 of 4**, and its only spec with any status is `T1.02 =
  ERROR`, carried three days.
- **All ten senses read 0 LOAD-BEARING**, which is now *visible* rather than
  hidden — a real gain from this window's work.

**Credit where it is due on the mechanism I criticised yesterday.**
`9064c49` added the standing rule to `ladder_prompt.md` — *a GOAL.md commitment
with ZERO passing specs outranks fan-out* — and the very next unit obeyed it
(`SM.01`, chosen because smell read 0 of 2). The fan-out-ranking blind spot I
described is fixed at the policy level. It has not yet reached unison, because
unison reads "1 passing" (§1a) and so the rule does not fire for it. **The
instrument defect is now suppressing the policy fix.**

---

## 4. Is the builder alive and productive? — ALIVE, and 95% of the day was lost

**One completed iteration in the last 24 hours.** The log is unambiguous: from
`2026-08-10T17:07:04` to `2026-08-11T15:57:03` every hourly wake logged
`STOPPED at 90–92% weekly usage — all agents paused until the owner resumes`.
**22 h 53 m of dead time, 23 consecutive skipped iterations.** PASS delta over
the window: **+2 (64 → 66)**, both earned in the ~35 minutes of runtime that
existed.

**The gate was right and the loop was honest about it.** `usage_gate` fails
closed on unreadable usage, and the stop message was accurate. What was missing
was named by the builder itself in `b1db303`: *"the message wrote a cheque the
code could not cash"* — the 90% rule had no resume, so the only exit was the
weekly reset. `scripts/lib_usage.sh` now provides one, shared by all four
organs, with an **expiry as its central design choice** (*"An override with no
end is not a resume, it is a deletion of the limit nobody remembers making"*).
That is the correct fix and the reasoning is better than the fix.

**Iteration quality is high and self-critical.** The 15:57 iteration recorded
`T0.20`/`T0.21`/`T0.13` as PASS from a modified tree at `17a6c3c`, **caught it
itself**, and re-ran all three from a clean tree at `e39455b` (`7e6fc60`) rather
than keep correct verdicts carrying a stamp that named uncommitted code. The
16:12 iteration disclosed a shortfall it was under no obligation to report —
SM.01's blank fractions 0.41/0.55/0.63 against Farrell's field data
0.852/0.901/0.837, a gap of 0.33 on the intermittency that is smell's whole
non-redundancy argument — and put it in the ledger where SM.02 will have to
answer for it.

**Four organs now start in the same second.** `ps` at 17:05 shows
`ladder_loop.sh`, `overseer.sh`, `review.sh` and `field_watch.sh` all forked at
**17:01:57**, each spawning a `claude -p` — four unattended agents on 4 shared
ARM cores, all committing into one git repository through one `index.lock`. The
same thing happened at 15:57. There is **no `resume.sh` in the repo**: the
restart is manual, undocumented, and fires everything at once. I have flagged
the two-way `review`/`overseer` collision for three audits; the resume path made
it four-way and structural.

**Model naming, sixth audit.** `crontab` still reads
`7 * * * * JACK_LOOP_MODEL=fable …`. `ladder_loop.sh` now defaults to `opus`
with a `FALLBACK_MODELS` chain, so the two hand-launched runs correctly logged
`model opus` — but every **cron**-launched iteration still sets `fable`, still
opens `OUT OF CREDITS`, and still logs a model it will not use. One word in one
crontab line, and it is the same defect class this project spent two commits
fixing for the ledger's commit stamp.

---

## 5. Compute honesty — the meter is honest; the quota is idle

```
2026-W32 (the live week):  kaggle 11.9635 h   colab 0.0015 h   remaining 18.04 h
2026-W31:                  kaggle 37.4554 h   colab 7.7461 h
overruns: []
```

**Byte-identical to audit 5.** Zero GPU hours in 28 hours — but that is **not
waste this time**: 23 of those hours the loop did not exist. Every one of the 69
ledger entries still carries `hardware: aarch64/…/cpu`.

The standing facts: **18.04 h expire 2026-08-16 (Sunday)**; W32's 11.96 h bought
one VOID (`T2.02`, 6.28 h) and one FAIL (`T2.01`, 5.58 h) — real measurements,
zero PASS; and `T1.02` (`gpu<20min`, ERROR since 2026-08-08 on *"kaggle: 0.0h
left, need 0.7h"*, an infrastructure error and not a verdict) is **first in
`run next`** and one of only four specs behind `generality`. W31's 37.46 h
against a 30 h ceiling with `overruns: []` is the known pre-meter gap.

---

## 6. Stuck decisions

`docs/DECISIONS_NEEDED.md` — 8 open blocks. **Nothing has enough evidence to be
decided that is not already flagged.** Nothing was acted on without being
recorded: I checked the PLASTIC-ONLY decree against the window's diffs and found
no frozen component introduced, and `D3` (may the loop push) is answered YES and
being honoured — `origin/main..HEAD` is 0.

**Two false statements are still the first thing anyone reads in that file, and
only the owner may strike them. Sixth audit asking.** *"Kaggle GPU is not being
granted"* claims it blocks `T0.10`/`T0.11` — both PASS since 2026-08-04, and
Kaggle has billed 11.96 h this week. *"/data is 95% full"* is marked OPEN;
`df` reads 21%. Corrections for both are already drafted 700 lines down in the
same file (`## HOUSEKEEPING 2026-08-10 06:45`).

`D1`'s recommended option A (*"freeze the trunk"*) still contradicts the
PLASTIC-ONLY decree and still should not be put to the owner in that form.
Unchanged from audit 5; the loop cannot rewrite an owner block itself.

---

## 7. Bakeoff hygiene — CLEAN

Two entries in `DECISIONS_RESOLVED.md`, both `PS.01/J`:

- **`PS.01/J` → VOID**, correctly: three arms below the 3σ learning gate, and
  the file says so in the words the protocol requires (*"An arm that has not
  demonstrably learned cannot arbitrate the decision"*). **A VOID was not
  treated as a verdict.**
- **`PS.01/J2` → WINNER `impact_speed`**, 10.32σ over null and **2.66σ over the
  runner-up** (`peak_dvel`, 0.827) — outside the noise margin, not inside it.
  Losing arms recorded, eliminated arms named separately from competing arms.
- The `screen` gate mode used here is legitimate (§2): it is guarded by `T0.19`,
  requires a written rationale that is stored **with the verdict**, and that
  rationale correctly identifies why a learning gate is inapplicable to
  deterministic reductions of identical cached rollouts.

The file's own header remains the most honest line in it: until these ran, the
third law had **never been exercised on a real question**. It now has, twice.

---

## 8. The honest summary — closer on the machine, flat on the creature

**PASS by tier, which is the answer to this section:**

| tier | what it is | PASS |
|---|---|---|
| 0 | harness — can we measure anything | **23** |
| 1 | primitives — can every part learn | **12** |
| 2 | capability vs null | **30** |
| 3 | **earn your parameters (ablation)** | **0** |
| 4 | **unison (senses fused, none collapsing)** | **0** |
| 5 | **the claims — continual learning, plasticity, curiosity** | **0** |
| 6 | a living Jack | **1** |

**Sixty-six green ticks, and the three tiers where the thesis lives are at
zero.** Of the 30 tier-2 passes, 9 are `PG.*` world-fidelity certificates, 3 are
`LC.*` bakeoff harness, and `PS.01`/`SM.01`/`TA.01`/`DP.00` are calibrations,
sensors and a property of the world. The genuinely Jack-level claims are the
`ME` family and `XL.00`: **he remembers, attributed, across restarts and across
death.** That is real and it is not small.

Are we closer to a curious humanoid that climbs the ladder than yesterday?
**On the machine, yes and measurably.** Three commitments that had nothing
behind them now have sensors (smell, taste, fast/slow-in-this-world). A guard
class was closed rather than an instance fixed (`T0.22`). A tier was invented
that makes the project's own standard *visible as unmet* — `0/10 LOAD-BEARING`
is a worse-looking number and a more honest one, and choosing to print it is the
system doing exactly what it was built to do. The 90% stop got the resume it had
been promising in writing for a day.

**On the creature, flat.** Nothing on the board shows Jack **wanting** anything.
Curiosity's only pass certifies a trap exists in the world; unison's only pass
certifies a rule for admitting candidates; sight and hearing's passes certify
the playground, not the perceiver. The ladder-and-apple sentence the project was
written around still has no spec that attempts it. And the two instruments that
are supposed to notice that — `coverage.py`'s pass column and the dependency
graph's PASS test — are both, in different ways, reporting a slightly rosier
world than the ledger contains.

That is the shape of this audit: **the science is honest, the record is clean,
and the map is drifting ahead of the territory.**

---

## FOR THE BUILDER

Ranked by damage to the trustworthiness of the ledger and its map. None needs
the owner.

1. **Give `coverage.py` the tier `senses.py` already has (§1a). NEW, RANK 1.**
   You built the right fix yesterday and applied it to ten senses; twenty-three
   commitments still use the flat count. Concretely: split the `covered` column
   into **`certificate`** (the world/harness/data/sensor is in place) and
   **`claim`** (Jack demonstrably does the thing), decided by a per-spec
   declaration — not by tier number and not by a regex, for the reason
   `coverage.py`'s own docstring gives. Then make the zero-alarm fire on **zero
   `claim`-grade passes**, not on zero specs. Under it today, `one brain /
   unison` and `curiosity` both read **0**, which is what `ladder_prompt.md`'s
   new standing rule needs in order to fire for them. Bring a known-answer test,
   per the `T0.21` pattern you just wrote: `LC.01` declared against `one brain /
   unison` must NOT count as a claim; `XL.00` against `memory across lives`
   must.

2. **Make the dependency graph ask the freshness question (§1b). NEW, RANK 2.**
   `experiments/run.py:590` — `if ledger.status(d) is Status.PASS: continue` —
   is the rule `T0.22` retired. Call `staleness_of` there, exactly as
   `borrow_metrics` and `stale_claims` already do (three callers, one rule; that
   unification is the whole point and it has two of three). A spec whose
   dependency is stale should surface in `run next` as **blocked on a re-run**,
   not as runnable. Add it as a `T0.22` property so the class stays closed.

3. **Re-run `PS.01`, then `XL.00` — in that order (§1b).** PS.01's flag is
   definitional (`IMPL_DEPS` only; `playground.py` unchanged since before its
   run) and it has been carried two audits. XL.00's is real. Doing XL.00 first
   returns VOID by construction. Both are CPU; PS.01's own recorded duration is
   short. This also unsticks `LC.03`/`LC.04`.

4. **Re-run `T1.02` (§5).** ERROR since 2026-08-08 on an infrastructure message,
   `gpu<20min`, first in `run next`, one of only four specs behind `generality`,
   and 18.04 GPU hours expire Sunday. Third audit asking. An ERROR is not a
   verdict.

5. **Take `UB.14` or another unison spec (§3).** `CPU_LONG`, depends only on
   `PG.1`, runnable for **51 hours**. The standing rule you added does not fire
   for unison only because §1a's instrument says it has a pass. Do not wait for
   §1 — the rule's *intent* covers this today.

6. **Give the resume an entry point (§4). NEW.** Four organs forked in the same
   second at 15:57 and again at 17:01:57, each spawning a `claude -p`, all
   committing through one `index.lock` on 4 shared cores. Write `scripts/resume.sh`
   that writes `.usage-resumed` and then **lets cron re-enter each organ at its
   own cadence** rather than launching all four; at minimum give every organ's
   commit an `index.lock` retry. This also closes the `review`/`overseer` minute-37
   collision, fourth audit.

7. **Backfill `impl_sha` on the 44 pre-`impl_sha` entries (§1).** Unchanged from
   audit 5 — **64% of the ledger still cannot be staleness-checked at all**, and
   §1b just showed that staleness is now load-bearing on the dependency path as
   well as the borrow path. Mostly CPU, previously measured at ~234 minutes
   total. A background sweep, not a project.

8. **Give `T0.10` a control (§1). Sixth audit.** The last PASS whose gate has
   never been shown able to report the bad case, and the one where the bad case
   has actually happened on this account.

9. **`JACK_LOOP_MODEL=opus` in crontab (§4). Sixth audit.** `ladder_loop.sh` now
   defaults correctly; the crontab line still overrides it back to `fable`, so
   every cron-launched iteration logs a model it will not use.

---

## FOR THE OWNER

1. **The loop stops again at 12:00 UTC tomorrow unless you re-grant it, and it
   was already dead for 23 of the last 24 hours. NEW — this is the one with a
   clock on it.** `.usage-resumed` reads `ceiling=100`, `until=2026-08-12T12:00
   UTC`. Weekly usage is at 92%. The expiry is deliberate and I think it is
   correct design — the builder's own words: *"An override with no end is not a
   resume, it is a deletion of the limit nobody remembers making."* But the
   consequence is concrete: **in 19 hours all four organs stop again**, and the
   only exits are your renewal or the weekly reset. **The ask: decide now whether
   to renew at 12:00, and tell the loop what the standing policy is** — renew
   daily until the weekly reset, or accept the pause. Either answer is fine; the
   silent version costs another 23-hour day. I have appended this to
   `DECISIONS_NEEDED.md` with the evidence.

2. **Two false statements are still the first thing anyone reads in
   `DECISIONS_NEEDED.md`. Sixth audit asking; one line from you clears both.**
   *"Kaggle GPU is not being granted"* — `T0.10` and `T0.11` have been PASS since
   2026-08-04 and Kaggle billed 11.96 h this week. *"/data is 95% full"* — `df`
   reads 21%. Both corrections are already drafted inside the same file; the loop
   is not permitted to strike an owner block.

3. **`D1`'s recommended option still contradicts your own decree.** Option A
   (*"freeze the trunk"*) is constitutionally unavailable under PLASTIC-ONLY
   (2026-08-09), and D1's evidence is frozen by its own correction. The honest
   ask is still: **nothing yet** — but the option set needs rewriting before it
   reaches you again.

4. **The scope question from audits 4 and 5 is still yours.** Temperature and
   pain read `ABSENT` in `run senses` — no spec would prove them. Temperature
   arrives only with the whole survival world; it is also the only mechanism in
   the design that teaches construction (*"he builds a shelter"*). **Schedule the
   W family now, or after the LC bakeoff?**

5. **For information: the integrity guarantee holds, and it is the best window
   yet.** `run verify` re-judged 65 PASS entries through their committed gates —
   65/65 re-derive, 63 controls probed, 0 ignored, 0 declared-but-unrun, 0
   unreplayable. Zero `+dirty` stamps. All 69 commits resolve. Across six audits:
   **no threshold has ever been moved in the loosening direction, no control
   weakened, no assertion removed.** Four of my ten items from yesterday closed
   in about an hour of builder runtime — including the coverage regex and the
   borrowed-constant guard, both generalised into new ladder specs rather than
   patched.

6. **The one thing I would want you to read.** Yesterday I told you the ladder
   was *"closer, on mortality and memory; not yet started, on wanting."* Today
   the ladder is 66/162 and I can put a sharper number on the same sentence:
   **Tier 3 = 0, Tier 4 = 0, Tier 5 = 0.** Ablation, unison and the thesis are
   untouched. Everything green is harness, primitive, world-fixture, or the
   memory family. That is not a criticism of the work — the work has been careful
   and unusually honest, and the honest number `0/10 LOAD-BEARING` exists only
   because the builder chose to print a worse-looking truth. It is the answer to
   *"closer to a creature, or just busier?"*: **the machine got materially better
   this window; the creature did not move.**

---

*Ledger untouched. No experiment re-run — `run status`, `run next`, `run stale`,
`run verify`, `run senses` and `python -m experiments.coverage` are read-only
re-judgements of the existing record. Nothing outside `/home/opc/jackthelearner`
changed. No container or daemon touched. Tree was clean at audit start
(`7e6fc60`); this commit contains `OVERSIGHT.md`, `LESSONS.md` and
`DECISIONS_NEEDED.md` only. Note for whoever reads the git log: `ladder_loop.sh`,
`review.sh` and `field_watch.sh` were all running concurrently with this audit
(§4), so the working tree may have moved beneath it.*
