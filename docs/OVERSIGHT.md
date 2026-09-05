# OVERSIGHT — 75th audit, 2026-09-05 18:37–19:0x UTC (at `4e21b94`)

## VERDICT: DRIFTING — the board was not empty today. `SO.08`, the project's only fresh dispatch and its only startable claim about Jack, has been refused since **04:20** by an admission estimate of **54,000 s** that nothing measured — while the spec imports no world, steps no physics, and is built from three dependencies that measure **1.4 s, 1.9 s and 60.8 s**. Fourteen builder slots reported the board empty and stood down correctly against that number. The box spent the day at **load 0.05** with **48,260.74 s** of budget unspent and **40 specs foreclosed, 100% of them one class**.

The clean part first, because it is large and it was checked mechanically.

**Section 1 — ledger integrity: clean, fourth consecutive audit.** All **104**
PASS rows of **139** resolved: **0** whose `commit` fails `git cat-file -e`,
**0** with no `control` declared in the registry. Two rows carry empty
`control_metrics` — `T0.01` and `T0.10` — and both declare
`"NONE, BY DECISION (52nd audit B5)"` with the reason on the spec (an import
either raises or it does not; an external service returning real artifact bytes
is its own falsifier). Pre-registered, not silent. **No finding.**

**Section 2 — thresholds and controls: clean.** Seven-day `git log -p` over
`registry.py`, `registry_expansion.py` and `experiments/tests/`, reading every
removed line carrying a threshold, a seed count, an assertion or an `or`. Every
removal in the window is a **refactor with its replacement in the same diff** —
the `_per_seed` / `_MEMO` helper block moved, `review_queue` imports re-sorted,
control-name property sets extended. **No numeric threshold moved in the
loosening direction, no control was deleted or weakened, no `_check` gained an
`or`, no seed count was reduced.** The two largest spec commits of the window
(`SO.07` `2dd3687`, `SO.08` `7af006e`) both *add* controls — `SO.08` carries a
donor-shuffled leg and an equal-donors leg, and `SO.07` carries `C-MISPLACE`
rate-matched from measured drop counts. **No finding.**

**Sections 6 & 7 — decisions and bakeoff hygiene: no finding.**
`decisions --check` rc=0. **0 UNDECLARED, 0 MEANS-ESCALATED, 0 OVERDUE, 0
UNROUTED-OWNER-ASK, 0 VANISHED-OWNER-ASK.** Nothing sits on the owner's desk
that a measurement could settle, so there is nothing to seize under rule 3 and
nothing to arm. Both live `PROGRESS.md` owner-asks are attributed (`#1`→`D23`,
`#3`→`D21`). `D15`/`D16` come due today and `D21` is the flagged same-day race;
all three are correctly ordered in tonight's handoff (below). `champions
--check` rc=0, every ratchet class at baseline. `review-queue` rc=0, **0
violations**. `coverage` exits **2** on the standing routed reds: `claim_dead`
4, `goal_unrunnable` 7 (of which `GEN.02`/`GEN.03`/`GEN.06`/`GEN.09` are the
four carried above a baseline of 3, routed as
`goal-cites-four-specs-that-resolve-to-corpses`, DUE 09-10), 3 empty classes.

**Section 4 — the builder is alive and it did not idle by choice.** 19 slots
fired 00:0x–18:0x, zero `PACING:` skips. From 04:20 onward every slot journals
the same re-derivation — *"SO.08 hold re-derived: 48,261 s < 54,000 s
worst-case"* — computed from `cpu_budget.json` by its own read rather than
inherited. The arithmetic is correct at every step. The builder is not the
fault here; it is the organ that reported the fault most honestly and most
often.

**Section 5 — compute honesty: no finding on spend, one on non-spend.** CPU
today **9,339.26 s of 57,600**, of which **9,205.09 s (98.6%)** is `SO.07`'s
registered claim run that VOIDed honestly at its pre-registered reference lane.
The remaining 134.17 s across four certificate re-buys is not the constraint and
never was. GPU: `2026-W35` holds **kaggle 18.9304 h of 30** — **~11.07 h expire
at 00:00 tonight**, the third consecutive week of large expiry (W33 ~22 h, W34
~28 h). That expiry is *correct*: `W36`'s named buyer is `D1.0`'s successor at
~16 h, gated on the two `d10-*` rows owed by tomorrow's FULL. Unspendable
inventory, not idle uptime.

---

## FINDING 1 — the only claim about Jack on the board was foreclosed for fourteen hours by a number nobody measured

**What is true, measured live at 18:37 (read-only; I ran no spec).**

| | |
|---|---|
| box load | **0.05** (1-min, 4 cores) |
| CPU budget remaining today | **48,260.74 s** of 57,600 |
| `SO.08` admission estimate | **54,000.0 s** — provenance `ENUM (no recorded duration to project from)` |
| refused? | **yes**, since the day passed 3,600 s of billing |
| specs foreclosed right now | **40** — and `Counter` over their budget classes returns **`{'cpu<2h': 40}`** |

`coverage` names `SO.08` as the **only FRESH dispatch at any cost class** and
prints it under *"QUEUE DEPTH — dispatchable TODAY"*. `gate_cpu_child` refuses
it. Both are running correctly; they disagree because they are keyed to
different facts, and the builder has reconciled them by hand in every journal
entry since 04:20.

**Why the 54,000 s is not a measurement of anything.** `child_estimate_s`
returns `min(enum, 4 × measured + 10)`, and for a spec with no ledger row there
is no `measured`, so it returns the class enum whole:
`BUDGET_SECONDS["cpu<2h"] (9000) × 3 seeds × 2 = 54,000 s`. `SO.08` has never
run. So the number gating it is the **child-KILL allowance for the largest CPU
class** — which `cpu_budget.py`'s own docstring measures at a **median 257×**
the true cost across 108 runner-lane specs, and names `LG.02` at **1.9 s against
54,000 s** as its worked example.

**`LG.02` is not a coincidence — it is the spec `SO.08` is built from.** The
registry instructs it in capitals: *"SHARES LG.02's MECHANISM AND MUST SHARE ITS
IMPLEMENTATION — one Laplace posterior over verified outcomes, imported by both,
never re-derived."* I read the implementation. `so_08_whose_hands.py` imports
`json, random, sys, tempfile, pathlib`, then `protocol`, `registry`, `plants`,
`lg_02._trust` and `EpisodicMemory`. **There is no world import, no survival
import, no physics step.** The run is 240 rounds × 12 lives (4 modes × 3 seeds)
of a Laplace posterior update over a 30-wide window plus a diary write. `DT =
0.2` is a comment about the cadence malaise is felt at, not a step loop.

Set the estimate against every spec it descends from:

| | class | enum | **measured** |
|---|---|---|---|
| `ME.9` (dep) | cpu<10min | 10,800 s | **1.4 s** |
| `LG.02` (dep, mechanism shared) | cpu<2h | 54,000 s | **1.9 s** |
| `SO.06` (dep, same family, same week) | **cpu<10min** | 10,800 s | **60.8 s** |
| `SO.07` (sibling — *does* step the world) | cpu<2h | 54,000 s | 9,201.5 s |
| **`SO.08`** | **cpu<2h** | **54,000 s** | **— never run** |

`SO.06`, registered in the same commit by the same author four days ago and the
spec `SO.08` imports its `Hand` from, is classed **`cpu<10min`** and measured
**60.8 s**. Had `SO.08` carried its own family sibling's class, its estimate
would have been **10,800 s against 48,260.74 s remaining — admitted at every one
of the fourteen slots since 04:20**, and the only startable claim about Jack
would have settled today instead of being scheduled for a midnight window.

**Where the class came from.** `git log -S` puts it at `b6518dd`, 09-04 19:15 —
the registration commit, **nine hours before the implementation existed**
(`7af006e`, 09-05 04:20). It has never been revised. It is a guess made before
there was anything to measure, and nothing in the loop re-examines a guess like
that once it is in the registry.

**The current classing is worse in both directions, which is why this is not a
loosening request.** `cpu<2h` on a seconds-scale spec (a) foreclosed it for a
day, and (b) buys a hung run a **15-hour** kill window. A class that matches the
measured cost refuses fewer legal runs *and* kills a runaway sooner. The day
ceiling, the load gate and `_run_isolated`'s kill are untouched by it.

**What I am NOT claiming.** I did not run `SO.08` — I may not, and it would have
billed the slack it needs. I have not measured its cost; I have shown that
**nothing has**, and that every measured thing it is made of is three to four
orders of magnitude cheaper than its gate. The repair is a measurement, not a
number I pick.

**Why no organ caught it.** The generic shape *is* routed —
`cpu48h-class-self-forecloses-the-day-meter`, DUE 09-08, amended 09-04, which
states the arithmetic exactly and reports `n_foreclosed_unmeasured = 36`. But
that row concludes: *"Why it is not a builder fix: every repair raises or splits
a tenant-protection ceiling (SYSTEM.md law 4), and the honest cheap alternative
— projecting a never-run spec from a CLASS prior — is a genuine estimator design
with its own bakeoff, not a constant edit."* **That sentence is false, and it is
the load-bearing error.** There is a third repair that raises no ceiling and
needs no estimator: *a spec whose declared class does not match its measured
cost is misdeclared, and the fix is to measure it and declare correctly.* The
row enumerated four policy options (i)–(iv) and never considered that a
particular declaration might simply be wrong. Having concluded "not a builder
fix", it parked the question behind a desk whose own instrument reads **drain
UNBOUNDED** — and the project's only live claim went with it.

**Bounded damage, stated honestly.** At 00:00 the day resets to 57,600 s ≥
54,000 s and `SO.08` is admitted; tonight's handoff (`scripts/ladder_prompt.md`
557–591) is correctly armed with the 74th audit's corrected ordering — `D21`
(the race) and `D16` first, **then start `SO.08`**, then `D15` and the `T0.33`
re-buy, because `D15`'s default stales the very CPU certificate the admission
rests on. That block also correctly warns that the 3,600 s is *billing slack,
not wall clock*. So this cost one day, not the spec. It cost the day on which
the project had exactly one thing to say about Jack.

## FINDING 2 — `coverage` prints "dispatchable TODAY" for a spec that cannot be dispatched today

`coverage`'s queue-depth block is the first instrument this organ is told to
read, and it printed `cpu<2h  3  LF.01, SO.07, SO.08` under *"dispatchable
TODAY (runnable, implemented, tracked, unparked, unsettled)"* at the same
moment `gate_cpu_child` was refusing `SO.08`. The parenthetical is an honest
enumeration of its criteria and CPU affordability is not among them — but the
words above it are **"dispatchable TODAY"**, and for fourteen hours they were
wrong about the one row that mattered. Every slot had to re-derive the
contradiction by hand from `cpu_budget.json`. Ranked below Finding 1 because
nothing false reached the ledger, and because the builder caught it every time.

## FINDING 3 — today's Review died at max turns, and its page steered the builder all day

`review.log`: `2026-09-05T06:37:04 review start — 20m / 60 turns` →
`Error: Reached max turns (60)` → `06:52:20 sealed docs/PROGRESS.md as an
INCOMPLETE RUN draft (rc=1)`. The banner is explicit that *"any verdict, any
section claiming 'no findings', and any instrument table"* below it is
**UNVERIFIED** — and `FOR THE BUILDER` is below it. Fourteen builder journal
entries today cite **"no unit manufactured per PROGRESS FTB 2"**: a full day of
stand-down authorised by a page the sealing machine marks unverified. No organ
reads the seal.

I am reporting this rather than routing it because **the loop already repaired
both halves today, unprompted, and I verified the commits.** `2302b52` (13:15)
makes `seal_output` sweep a dead run's *whole dirty set* rather than the report
alone — the 74th audit's finding that the dead run's four other edits went out
unmarked — and raises `TURNS_PER_MIN` 3→6 on measured consumption (4–5.5/min;
7/7 max-turn deaths left wall time unspent, 4/4 cron FULLs died at turns). That
is the right diagnosis: an organ budget, not a science threshold. **What remains
unowned is the composition** — nothing refuses to *act* on a sealed page. The
stand-down happened to be correct on its own re-derivation, so this cost
nothing today. It is a live path for a dead run's draft to steer 24 slots.

---

## FOR THE BUILDER

1. **Size `SO.08` and declare the class its measurement supports — this is the
   highest-value hour available to you and it is not blocked by anything.**
   Take a SIZING RECORD the way `DP.04` and `LC.07` did: time `_measure(0)` at
   full `N_ROUNDS=240` for one mode, one seed, by hand (`cpu_budget.py`'s own
   scope note: *"A module invoked BY HAND remains unmetered"*), record the
   seconds in the commit, then set `budget=` to the class that number supports
   and re-buy the certificates that cite the registry. **Do not pick the class
   from my table — measure it.** If it lands above `cpu<10min`, say so and keep
   `cpu<2h`; a refuted expectation recorded is worth as much as a confirmed one.
   Two guards, because this admits runs that are refused today: the child-kill
   timeout **tightens** under any lower class, and `T0.33`'s
   `projection_only_tightens` / `est_above_enum` properties must stay green in
   the same commit.
2. **Tonight's 00:0x ordering is correct as written and I verified it — do not
   re-derive it under time pressure.** `D21` (the same-day race, before 06:37)
   and `D16`, then **start `SO.08`**, then `D15` + the `T0.33` re-buy. The
   3,600 s is billing slack, not wall clock. If item 1 lands first, `SO.08` is
   admissible *now* and the midnight race stops being a race at all.
3. **Add the affordability fact to `coverage`'s queue-depth print (Finding 2).**
   Not a new gate and not a new ratchet — the block already prints `HELD by an
   open decision` and `NOT FILLABLE` annotations, so this is one more
   annotation in an existing idiom: mark a row `UNAFFORDABLE TODAY (est Ns vs
   Ms remaining)` when `gate_cpu_child` would refuse it, and subtract it from
   the FRESH count. `coverage` must not import a *decision* from the meter,
   only the estimate and the remaining seconds.
4. **When you route a defect, do not also rule out a class of repair you did not
   test.** The amendment on `cpu48h-class-self-forecloses-the-day-meter` closed
   the door on a builder fix on a general argument about ceilings, and a
   per-spec misdeclaration walked straight through it. If a routed row says
   "not a builder fix", the reason must be a measurement or a rule citation,
   not a plausibility argument.
5. **Standing prohibitions unchanged and re-affirmed:** do not re-dispatch
   `D1.0`; `HR.1`–`HR.4` stay `D19`-held to 09-14; `HR.6` stays behind `HR.5`;
   `LF.01` attempt 2 waits for the 09-09 design; do not re-stagger the 09-06
   docket by hand. Item 1 above is a *cost declaration*, not a threshold, and it
   is the only spec-file edit this audit asks for.

## FOR THE OWNER

1. **NO-DECISION — a finding you should see before you rule on `D22` (due
   09-08), because it changes one of that entry's premises.** `D22` asks whether
   design authority should stay solely with the Review, and its stated cost is
   *"all 9 startable specs behind the desk, and 0 FRESH dispatches at any cost
   class."* Today there was a tenth, `SO.08`, and it was behind **neither** the
   Review nor the owner nor compute scarcity — it was behind a one-word budget
   class guessed at registration, on a box idling at load 0.05 with 48,260 s
   unspent. That is not an argument for or against `D22`; it is a correction to
   the picture the entry paints. **A share of what this project reads as
   "blocked on design" is blocked on bookkeeping nobody re-examines**, and the
   two have opposite repairs. I have asked the builder to measure and re-declare
   (B1) rather than routing it to the desk `D22` is about, on purpose.

2. **NO-DECISION — the drain, unchanged and worsening, reported because it is
   the standing number.** `review-queue` rc=0 with **0 violations** and
   **38 live rows of 41 routed**; trailing 7 days, arrivals **5.14/cycle**,
   disposals **0.29/cycle**, **drain UNBOUNDED**. Six rows share tomorrow,
   2026-09-06, against a measured capacity of ~1/cycle, and tomorrow's FULL also
   owes Part 2 and both completeness audits. The queue reports zero violations
   *by construction* — its classes fire on a promise breaking, not on a promise
   being unpayable — which is exactly the composition `D23` (due 09-11) exists
   to make into a printed integer. Nothing new to rule on; `D22` and `D23` are
   already yours and already dated.

3. **NO-DECISION — liveness, verified against `/data/jack-logs` mtimes rather
   than any organ's report.** Builder `ladder.log` 18:10, 19 slots today, zero
   `PACING:` skips. Overseer `overseer.log` 18:37 (this run). Review
   `review.log` 06:52 — **died at max turns, sealed, and the turn budget was
   repaired at 13:15 in `2302b52`**; next fire is tomorrow's FULL at 06:37.
   Field watch 08-31 05:53 (Mondays; next 09-07, inside cadence).
   `lost_iterations.log` still 0 bytes. No organ is silent.

4. **The honest paragraph (section 8): are we closer to a curious humanoid, or
   only to a longer list of green ticks?** Today, neither — and for the first
   time in a while the reason is not that the science was hard. The ladder moved
   **+0 demonstrated** in the 24 h to this audit. Every ratchet is at or inside
   its baseline; no threshold moved; the ledger is clean; four instruments read
   green and the fifth is red on reds it correctly refuses to forget. The
   machine is in excellent health and it manufactured nothing, because the one
   unit that would have asked a real question about Jack — *does his diary
   record whose hands left the gift, and does he act on it* — sat implemented,
   gated, controlled three ways, and refused since breakfast by an estimate
   28,000× the measured cost of the mechanism it borrows. **Fourteen slots
   reported an empty board and all fourteen were telling the truth as their
   instruments gave it to them.** That is the shape worth your attention: this
   project's failures have stopped being wrong answers and become correct
   answers to questions no one noticed were mis-specified. The `SO.07` VOID
   yesterday was the world refusing to produce a behaviour — real science,
   honestly recorded. Today's stall was arithmetic. The first kind is the cost
   of doing this work; the second is not, and it is the cheaper one to fix.
</content>
</invoke>
