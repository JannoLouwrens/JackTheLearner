# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-09 06:37 UTC — 147th audit.**

## VERDICT: DRIFTING

**The ledger is clean and the perishable resource is already lost — not
"at risk", lost, by arithmetic that can be done today.** `2026-W40` has
**28.03 of 30 free Kaggle GPU-hours undrawn and they expire Saturday
2026-10-10**. The builder is the only organ that can spend them, and
`pace_gate` cannot legally release it until the usage week is **64 % elapsed**,
which is **≈ 2026-10-12 20:30 UTC — about 2.4 days AFTER the hours die.**
That is not a forecast of a risk; it is a closed arithmetic, and it is the
first time this loss has been computable *before* the expiry rather than
reported after it. Fourth consecutive week; ~114 h unbought in four.

**Section 1 is clean and that is the valuable half.** All **107** standing PASS
rows resolve in `BY_ID`; every one names an implementation that exists on disk;
every `commit` resolves in git; and **every PASS that declares a control has
`control_metrics` recorded — 0 exceptions.** Only `T0.01` and `T0.10` declare no
control at all, both harness smoke rows, both already owned at
`t018-explicit-no-control-reads-as-an-unrun-promise` (OPEN, DUE 2026-10-19).

**Section 2 is clean.** In seven days **not one numeric bar moved in the
loosening direction.** Ten commits touched `registry*.py` or `experiments/tests/`;
every constant in them is an **addition** (`W1.01`, `W1.04`, `T4.04`, `T4.05`,
`T6.01` newly implemented; `T1.08`'s `TAIL_FRAC`/`WARMUP`), and the one change to
an existing spec's gate — `T1.11`'s `SHIPPED_CALLER_MIN = 1` — **tightened** it
into a FAIL. There is no `-CONST = N` / `+CONST = M` pair anywhere in the window.
`MAX_HELDOUT_CV_PCT` 7.0 is byte-unmoved across all three `T1.08` attempts,
verified at source.

**Instruments, re-run immediately before this file was committed** (the Review
is in flight in the same 06:37 slot, so a reading quoted from the top of a
sitting is the default failure here): `coverage` **2**, `decisions --check`
**0**, `champions --check` **0**, `run status` **2** (unpiped), `run
review-queue` **2**. Floors: **3 ABOVE** (`dark_slots` 18 vs 0,
`pass_on_dead_dependency` 6 vs 3, `unreachable` 96 vs 95), 0 BELOW,
0 UNVERIFIED. None of the three reds is new.

> **Concurrency disclosed:** `review.sh` (pid 1879250) started at 06:37:03
> alongside this audit and committed `cff86c3` at 06:43:25 during it. Its work
> is credited below rather than re-discovered, and `experiments/ledger.json` /
> `experiments/cpu_budget.json` were dirty in the tree from that sitting while I
> wrote. **I staged only `docs/OVERSIGHT.md` by name.**

---

## RANK 1 — the W40 GPU quota is unreachable by arithmetic, and three
## independent locks each sit in a different instrument with no reader joining them

Every number here is derived from source, not quoted.

**Lock 1 — the clock.** `scripts/lib_usage.sh:85`:

```
allow = PACE_FLOOR + ((PACE_CAP - PACE_FLOOR) * elapsed + 99) / 100
PACE_FLOOR=25   PACE_CAP=90        # skip iff  pct >= allow   (line 86)
```

Live now: `week:all models` **66 %**, `--week-elapsed` **13**, so
`allow = 25 + ceil(0.65 × 13) = 34`. 66 ≥ 34 → skip. Release needs
`allow ≥ 67`, i.e. `elapsed ≥ 64`. The meter resets **Oct 15 09:00 UTC**, so the
week opened **Oct 8 09:00 UTC** and 64 % of 168 h lands at
**≈ 2026-10-12 20:30 UTC.** The Kaggle hours die **Saturday 2026-10-10.**
**The gap is ~2.4 days and it is not closable by waiting.**

**And ≈Monday 20:30 is a FLOOR, not an estimate — it assumes the meter freezes
at this instant.** The line rises at a constant **0.3869 pts/h** (the function's
own comment). The meter rose **63 % → 64 % → 66 %** between 05:07 and 06:55
this morning — **~1.7 pts/h, 4.3× the line's rate.** While the meter climbs
faster than the line, the release date moves *away* faster than the clock moves
toward it. Each additional meter point costs **~2.58 h** of further delay.
**Of this week's 37 shared points: builder 8 (21 %), desks 4 (10 %), NOT THIS
PROJECT 25 (67 %).** This project is being paced out of its own free GPU week by
another tenant's spend, and at this morning's rate the gate may not release
before the usage week's own reset on Oct 15 — by which time W40 is five days
dead.

**Lock 2 — the buyer is spent, as of 06:43 this morning.** The 146th audit's one
piece of good news, written 12 hours ago, was *"Unlike the previous three, W40's
hours have a buyer that is buying."* **That sentence stopped being true at
06:43:25 today.** `cff86c3` stamped `t108-step-2b-recipe-repair-is-routed-off-
eval-cv-0-52` **ACTED**: both mechanisms implemented, dispatched, harvested, and
the row's own pre-registered branch fired — *"the residual is NOT a third
mechanism — the row forbade that in writing."* W40's named buyer is closed, and
the successor row `t108-recipe-repair-is-exhausted-and-the-attribution-pair-
flipped-sign` states in terms that it has **"no GPU buyer at all"** and dates
itself **2026-10-21**, off the expiring week on purpose. **That reasoning is
correct and I am not arguing with it** — buying another 3-seed reading of a
statistic under suspicion is the one purchase the evidence forbids.

**Lock 3 — there is no fresh GPU dispatch at any GPU cost class, awake or not.**
From `coverage`'s QUEUE DEPTH, re-read this sitting:

| class | dispatchable | state |
|---|---|---|
| `gpu<20min` | **0** | EMPTY — pilot BLOCKED on evidence (`DP.04`, `SM.03`); repair is a REDESIGN |
| `gpu<2h` | 1 (`UB.10`) | VOID — "no FRESH dispatch here" |
| `gpu<8h` | **0** | EMPTY — pilot BLOCKED on evidence (`LC.07`); repair is a REDESIGN |

Headline: **"dispatchable TODAY: 7, of which 7 VOID → only 0 is a FRESH
dispatch."** So even a builder released this hour has nothing to spend the quota
on.

**Why this is a finding and not three re-reports.** Each lock is visible to
exactly one instrument — the clock to `ladder.log`'s PACING line, the buyer to
`review-queue`, the empty classes to `coverage` — and **no reader in this repo
joins them.** The joined reading is the one that matters and it is not available
anywhere: *the Kaggle quota has stopped being a constraint on this project at
all.* The binding constraints are a shared usage meter two-thirds spent by
somebody else and a dispatch queue with no fresh GPU work in it. `D30`'s
standing report and `D40` both still frame the loss as hours going unbought
through negligence or bad luck; it is now over-determined, and reporting
"~28 h expired again" next week will describe the least important of three
causes.

## RANK 2 — the one real experiment of the week moved the number the WRONG WAY,
## and the instrument defending a Tier-1 bar has contradicted itself

Credit where it is owed: **the Review found this independently at 06:43 and
routed it well.** I verified it at source rather than inheriting it.

| attempt | commit | `heldout_cv_pct` | final-iterate | mechanism |
|---|---|---|---|---|
| 3 | `3d357c4` | 40.006 | — | none |
| 4 | `381a9e6` | **14.666** | 16.994 | (i) tail-average |
| 5 | `b7aa2bf` | **17.285** | 16.861 | (i)+(ii) cosine-to-0 |

Bar **7.0**, byte-unmoved, re-verified at
`experiments/tests/t1_08_seed_variance.py:190`. Mechanism (i) is **unchanged
between arms 4 and 5**, same `spec_sha`, same T4 venue — and its attributed
effect reads **16.994 → 14.666** in one and **16.861 → 17.285** in the other.
**Opposite sign, same mechanism.** The pair built to attribute (i)'s effect
instead established that (i)'s effect is smaller than the noise of the estimator
measuring it: a 7.0 bar is being adjudicated by a statistic whose own
arm-to-arm swing is ~2.6 CV-points.

**The honest consequence nobody has written down: the tree is currently left
holding the WORST-measured of the three configurations.** `t1_08_seed_variance.py`
contains both mechanisms live (`TAIL_FRAC = 0.2` at :193, `WARMUP = 100` at :210,
`lr_schedule "warmup100+cosine_to_0"` at :405), so a sixth attempt run from HEAD
re-buys 17.285, not 14.666. That is legal — (i) was deliberately kept so arm 5
would attribute (ii) — but it means **the spec's shipped implementation is now
the configuration with the highest recorded dispersion, and no record says so.**
Routed below as FTB 1; it is a disclosure, **not** a licence to revert, because
reverting (ii) without a verdict is choosing a recipe on a statistic this
project has just measured as unable to choose.

This matters beyond `T1.08`: it is **the project's largest blocker (blocks 45
specs)**, and `T2.03` and `T2.14` are both standing PASS rows resting on it
(`pass_on_dead_dependency`).

## RANK 3 — `docs/PROGRESS.md` is five days old and its own staleness banner
## understates that by 73 hours

The Review's current-state page last had its content rewritten **2026-10-04**
(`8a42c10`, items edited in the 10-05 run that died `rc=124`). Today is 10-09:
the page is **~120 h old against a 25 h cadence.** Its banner reads *"last moved
47h ago"* and is stamped **2026-10-07T06:37:04** — because `lib_seal.sh` returns
early when a banner already exists, so **the staleness notice is itself two days
stale and understates the age by 73 hours.**

This is the same defect the 10-04 FULL diagnosed in its own banner (*"understated
that by 30"*), and **the number has more than doubled since.** Its routed repair
is that page's own **FTB 5** — stop writing the page last — which is a
`scripts/review_prompt.md` change the builder owns and the builder has been dark
for every slot since. `PROGRESS_LOG.md` now carries **nine consecutive
`INCOMPLETE` rows, 09-29 through 10-08**: every one a sitting that made real acts
and lost only its page. The desk is working; the record of the desk is not.

**Read with care, because the fix is cheap and the readings are not equivalent:**
the page's `FOR THE BUILDER` and `FOR THE OWNER` sections are the *2026-10-04*
asks, and I read them as live this audit (items 1–7 and owner items 1–6). The
`VANISHED-OWNER-ASK` counter reads **0/0** — the asks did reach `D40`/`D33`/
`D41`/`D42` and are attributed — so nothing has been lost. The damage is that
the project's one current-state progress page has been a record for five days
while four audits and five Review sittings read it as current.

## RANK 4 — the three ABOVE-floor ratchets, and one of them cannot measure
## its own worst reading

`dark_slots = 18` (floor 0), `pass_on_dead_dependency = 6` (floor 3),
`unreachable = 96` (floor 95). None is new and all three are owned. But one
reading is worth naming:

**`dark_slots` is a gauge, not a counter, so its peak is unrecordable.** The
committed reading is **0, recorded 2026-09-30**. Between then and now the
builder went dark for the longest streak in project history — the 10-04 FULL
measured **76 consecutive slots / 75.8 h**, and that page's FTB 0 told the
builder its last slot had ended **99 slots** earlier. The counter passed through
~99, returned to 0 when the builder woke on 10-06, and now reads 18. **The
ratchet's committed history shows `0 → 18` and the 99 is nowhere in it.** A
gauge whose floor is 0 is also ABOVE floor during *any* legal pacing, so its
contribution to "3 ABOVE" is partly a tautology rather than a finding. Reported,
not repaired: the fix is a high-water mark beside the live value, and it is the
builder's.

`pass_on_dead_dependency` 6 against floor 3 is the honest reading of "107
demonstrated": `LF.02 ← T6.03 BLOCKED`, `T0.18`/`T0.19 ← T0.13 FAIL`,
`T1.12 ← T1.11 FAIL`, `T2.03`/`T2.14 ← T1.08 FAIL`. Three of the six trace to
`T1.08` and `T1.11`, i.e. to RANK 2 and to the shipped-caller finding — **the
repair for half this counter is the same repair.**

## RANK 5 — `GOAL.md` cites four specs that resolve to corpses, and the
## count is flagged NEW

`coverage` reports **4 NEW unrunnable citation(s): `GEN.02`, `GEN.03`, `GEN.06`,
`GEN.09`** — every one `welded ← LC.07`, which is `PILOT-BLOCKED`. Each id
*resolves*, so the dangling-reference count cannot see it; and each is welded, so
the citation's present tense is false. This is the 59th audit's B2 class: *an id
that resolves to a corpse is a worse dangling reference than one that resolves to
nothing.* These are the GOAL.md sentences on learning-by-watching, false belief,
transfer across worlds, and culture — three of them the "OTHER MINDS" and "MORE
WORLDS" expansions the page calls the destination.

It is owned (`gen-four-revival-needs-an-affordable-lc07-successor`, OPEN, DUE
2026-10-11, and `gen-four-reparented-to-a-decision-that-had-already-closed`,
ACTED), so this is a ranking, not a new route. **The instruction on the counter
is the load-bearing part: `GOAL_UNRUNNABLE_BASELINE` is shrink-only and must not
be raised to absorb these four.**

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, no findings.** 107 PASS rows, 107 resolve
in `BY_ID`, 107 implementations present on disk, 107 commits resolve in git,
0 PASS rows with a declared control and no recorded `control_metrics`. Two PASS
rows declare no control (`T0.01`, `T0.10`), both already owned. Separately and
legally: 2 DIRTY STAMPS (`T6.03`, `PL.02` — both pre-date `dirty_files`),
13 STALE CLAIMS, and 6 UNBACKED CERTIFICATES, all reporting-only and all
unchanged from the 146th audit.

**2. Thresholds and controls over time — CLEAN, no findings.** See the header.
Ten commits in window, zero loosening, one deliberate strengthening that cost a
certificate. **Saying so plainly is the result.**

**3. Drift from the goal.** The builder's last productive slot (10-08 12:07,
`rc=0`, 25 min) did `T1.08` Step 2b — which traces to *"every capability claimed
only by an experiment that could have failed … at ≥3 seeds"*, and it is the right
unit: `T1.08` is the largest blocker on the board. **No drift in what was
worked on.** The converse question is where the damage is, and it is unchanged
from the 10-04 FULL's reading: **`claim_dead = 3`** — **smell**, **shelter/
building**, **thermal ("too cold kills him")** — each an owner commitment with
no live falsifier, each parked on honest evidence, each needing a successor spec
the ladder cannot generate from inside itself. Plus **14 commitments with live
claim specs and nothing passing**, including `curiosity` (2 pass of 12),
`one brain / unison` (1 of 28), `fast/slow` (0 of 8, five welded behind
`LC.03`), and `sleep` (0 of 5). `commitments_uncovered` is **0** — every
commitment has a spec; the hole is that the ladder's centre of gravity is still
its own apparatus.

**4. Is the builder alive and productive?** **Alive, legal, and locked out.**
Last `rc=0` **2026-10-08T12:32:55**; **17 consecutive dark slots / 17.6 h** at
06:07, 18 by the ratchet now; **0 failed slots.** Every skip since 13:07
yesterday is `pace_gate` acting correctly on `week:all models` against its line.
Before that, four slots (08:07–11:07 on 10-08) were lost to **session limits on
every model** — fable, opus and sonnet all refused — and were correctly marked
and inherited. **This is not a sick builder and nothing here is the builder's
fault.** See RANK 1 for what it costs.

**5. Compute honesty.** `2026-W40`: **1.971 h drawn of 30, 28.03 h expire
Saturday 2026-10-10.** All three W40 jobs are `ok:true` and all three are
`T1.08`. **Two per-job overruns recorded, both this week** —
`…1791444041` 0.6544 h billed vs 0.5 est, `…1791461828` 1.0458 h vs 0.5 est —
and the second is worth a note: `projections` logged **0.75 h** for that job at
12:17:04 while `overruns` prices it against **0.5**, so the overrun channel is
not reading the job's own projection. **Direction is safe** (it over-states the
overrun, 2.09× instead of 1.39×), which is why this is a note and not a rank.
`gpu_hours_no_verdict` TOTAL **51.47 h**; the standing waste is still one row —
**`D1.0`, 33.78 h across 2 attempts, 0 verdicts** — and `UNATTRIBUTED` 6.32 h
across 21 jobs sits at its floor.

**6. Stuck decisions.** `decisions --check` **exits 0 with its whole ratchet at
floor**: 0 `UNDECLARED`, 0 `MEANS-ESCALATED`, 0 `UNROUTED-OWNER-ASK`, 0
`VANISHED-OWNER-ASK`, 0 `default-action-expired`, and **31 of 31 firings
transcribed** onto `DECISIONS_RESOLVED.md`. **Nothing a measurement could settle
is sitting on the owner's desk, and there is no `UNDECLARED` entry to arm this
audit** — the standing instruction to arm one per audit has nothing to act on,
and I am saying that rather than manufacturing an entry. `D41` and `D42` are
armed with monotone defaults, both `decide_by 2026-10-18`. Three `CONDUCT-DESK`
entries are stale and are the desks' own to execute, not the owner's:
**`D33` (16 d — MOOT, its object went terminal when `w1-world-edit-window` was
`DECLINED`, so no desk can clear it by firing anything), `D35` (15 d),
`D38` (5 d).** `D38` is the Sunday collision between `D28`'s OVERDUE-first and
`D33`'s W1-design-first; it is the Review's to rule and it next binds on a
Sunday, so it is not overdue in effect until 2026-10-11.

**7. Bakeoff hygiene — no new findings.** `champions --check` **exits 0** with
all 10 violations **at their declared floors**; no seat lost a door this window.
The standing bad news is unchanged and is not re-litigated: the **Learning-core
seat is held BY VERDICT off a VOID** (`LC.03`) with all three re-open triggers
closed doors, and the **World seat is held BY VERDICT naming neither a deciding
row nor a trigger** — the strongest marking in the file, twice, backed by a
non-verdict and by nothing. 4 seats are unwinnable, 2 unfalsifiable.

**8. The honest summary — are we closer to a curious humanoid, or to a longer
list of green ticks?** Neither, this week, and the shape of the standstill has
changed in a way worth naming precisely.

We are not closer to Jack. The one experiment that ran moved its number from
14.666 to 17.285 against a bar of 7.0 — **away** — and the more useful thing it
produced was the discovery that the instrument adjudicating it cannot tell those
two numbers apart. That is a real result and the right desk routed it within the
hour, as an estimator question with the bar explicitly protected from being
lowered. A project that answers a negative result by auditing its own ruler
instead of trying a third recipe is a project whose scoreboard can be believed.

But we are not closer to a longer list of green ticks either, and that is new.
**`demonstrated` has not moved in a week: 107 → 107.** Of 76 settle events in
seven days, **6 were first-ever verdicts, 67 were re-buys, and 68 of 76 (89 %)
were instrument-coupled** — this project's own tool edits re-staling its own
certificates. Three Tier-0 instrument specs account for 58 of the 67 re-buys
(`T0.21`×20, `T0.28`×20, `T0.31`×18). The machine is now spending most of its
recorded activity proving that its measuring apparatus still works.

The thing I would put in front of the owner above everything else is not a
failure of honesty — the honesty is in excellent repair, and Sections 1, 2, 6
and 7 are genuinely clean. It is that **the resources this project depends on
have quietly stopped being ours to allocate.** Thirty free GPU-hours a week
expire untouched for the fourth consecutive week, and this week it is provable in
advance that no legal slot exists before they die. Two-thirds of the usage meter
that gates the only organ able to spend them belongs to another tenant. The
constraint on building Jack is no longer the ladder's difficulty, the builder's
competence, or this project's honesty. It is that the project is third in line
for its own compute, and no instrument in the repo says so in one sentence.

---

## FOR THE BUILDER

0. **EVERY ORDER FROM THE 135th–146th AUDITS IS STILL OPEN AND NONE OF IT IS
YOUR FAULT.** Your last `rc=0` was 2026-10-08T12:32 and `pace_gate` has refused
every slot since, correctly. Read those reports and `docs/PROGRESS.md`'s FTB 1–7
as live and unexecuted. I am re-ranking nothing. **When you do get a slot, note
that RANK 1 means a GPU dispatch is almost certainly the wrong unit** — there is
no fresh GPU work at any class, and the hours die Saturday.

1. **DISCLOSE, DO NOT REVERT: `T1.08`'s shipped implementation is the
worst-measured of its three configurations.** `experiments/tests/t1_08_seed_variance.py`
carries mechanism (i) (`TAIL_FRAC = 0.2`, :193) **and** mechanism (ii)
(`WARMUP = 100`, :210; `lr_schedule "warmup100+cosine_to_0"`, :405), which
together recorded `heldout_cv_pct` **17.285** — above attempt 4's 14.666 and the
highest of the two repaired arms. Add that fact to the spec's docstring so the
next reader cannot mistake HEAD for the better arm. **Do NOT revert (ii), do NOT
touch `MAX_HELDOUT_CV_PCT` 7.0, and do NOT dispatch a sixth attempt** — the
estimator question is routed to the Review as
`t108-recipe-repair-is-exhausted-and-the-attribution-pair-flipped-sign`
(OPEN, DUE 2026-10-21) and choosing a recipe before it is answered is choosing on
a statistic this project has just measured as unable to choose. Docstring only;
zero behaviour change; verify the three numbers at source before writing them.

2. **`dark_slots` NEEDS A HIGH-WATER MARK** (RANK 4). It is a gauge: it passed
through ~99 during the 10-01→10-06 blackout, returned to 0, and the committed
ratchet reading went `0 → 18` with the peak nowhere in it. Print a
`dark_slots_max` beside the live value, persisted, monotone-up, reset only by an
explicit recorded act. **Reporting-only — do not gate on it**, and do not change
`dark_slots`' own floor of 0.

3. **THE OVERRUN CHANNEL DOES NOT READ THE JOB'S OWN PROJECTION** (section 5).
`experiments/gpu_budget.json` `overruns[1]` prices `…1791461828` against
`est_hours 0.5` while `projections` logged **0.75 h** for that job at 12:17:04,
40 minutes before it ran. Make the per-job overrun read the most recent
`projections` entry for the same spec at or before dispatch. **Strictly a
reporting fix and the current direction is the SAFE one** — it over-states the
overrun — so this may not displace item 1, and it must not be allowed to reduce
any recorded overrun below what a correct reading gives.

4. **STILL UNDISCHARGED AND STILL THE CHEAPEST THING ON YOUR BOARD:**
`docs/PROGRESS.md` FTB 7 — `experiments/fieldwatch.py:102`, `\bfinding\b` →
`\bfindings?\b`. One character, strictly additive, zero staleness bill. It
perished at the ~10-12 Monday sweep per its own row, so verify the defect against
the live regex yourself and **stop and route if it no longer reproduces.**

## FOR THE OWNER

**1. ONE LEVER, AND IT IS YOURS ALONE: `.usage-resumed` does not exist on disk,
and writing it is the only thing that can reach W40's 28 free GPU-hours before
they die Saturday.** `scripts/lib_usage.sh:77` — `pace_gate` returns "proceed"
unconditionally when `$REPO/.usage-resumed` is present. Without it the builder
cannot legally run until the usage week is 64 % elapsed, **≈ 2026-10-12 20:30
UTC**, and the hours expire **2026-10-10**. **I am not recommending you write
it, and I want to be exact about why.** The file also governs the 90 % hard stop
you decreed on 2026-08-09, `week:all models` reads **66 % and rising ~1.7 pts/h**,
and RANK 1's Lock 3 says there is **no fresh GPU dispatch at any cost class** —
so releasing the builder today would spend your shared budget and still buy no
GPU work. The honest statement is that the lever exists, it is yours, and this
week it opens onto an empty room. **Already on your desk as `D40`
(`decide_by 2026-10-10`, default (v) = the status quo); nothing new is asked.**

**2. NO-DECISION — `D30`'s standing report, delivered as its armed default
requires, with nothing to rule on.** Builder dark **18 slots / 17.6 h**, 0 failed
slots, last `rc=0` 2026-10-08T12:32:55; before that, 4 slots lost to session
limits on every model. **`2026-W40`: 1.971 h drawn of 30; 28.03 h expire Saturday
2026-10-10** — fourth consecutive week, ~114 h in four. **New this week and the
reason RANK 1 exists: the loss is now over-determined** — the pace clock, an
exhausted buyer, and an empty GPU dispatch queue, any one of which alone would
cost the hours. Of 37 shared usage points: **this project 12 (32 %), another
tenant 25 (67 %).** `demonstrated` 107 → 107. All four organs fired within
cadence.

**3. `D41` and `D42` are armed and both fall due 2026-10-18; this is a pointer,
not a re-ask.** `D41` — which artefact is the thing we are building, the ladder's
rig or `TrainingPipeline.py`; default (iii) NEITHER, monotone. `D42` — which ONE
of your three claim-dead commitments (**smell**, **shelter/building**, **thermal**)
gets a successor spec; default (iv) HOLD, monotone. Both defaults are
deliberately not the recommendation, because a default may not decide what the
project is or pick among your own commitments. **The fifth consecutive report to
carry `claim_dead = 3`, and the third to carry it somewhere it can be answered.**

**4. NO-DECISION — what this audit did not do, named rather than omitted.**
I arm no decision this sitting because `decisions --check` reports **0
`UNDECLARED`** — there is nothing to arm, and inventing an entry to satisfy a
per-audit quota would be the deadlock it was written to prevent, wearing a
process. I also did not re-derive the `champions` or `coverage` completeness
audits: both tools exit at their declared floors and the cognitive-half hole is
owned with a clock (`completeness-audit-2026-09-13-the-cognitive-half-is-the-hole`,
OPEN, DUE 2026-10-11). The Review's own 2026-10-09 armed stop-rule — the one
that DECLINES the orphaned `w1-*` class to you as a class — falls due **today**
and that sitting was still in flight when I committed; it is theirs to fire, not
mine, and the next audit should check that it did.
