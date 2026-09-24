# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-24 00:37–01:0x UTC — the 111th audit.** Six hours after the 110th
(18:37 yesterday). The window is the builder's six live slots 19:07–00:07, nine
commits (`955b9ef`, `eca5757`, `73c49a8`, `3e9cf42`, `92cb7bd`, `5b18cd3`,
`1ab5484`, `b0c7c45`, `308e202`) and two ledger events, both re-buys (`T0.36`
attempt 20 PASS 20:15, `T0.21` attempt 22 PASS 00:14). Every instrument here was
re-run against the tree immediately before committing.

---

## VERDICT: ON TRACK — with one standing PASS certificate unpaid, because the instrument built to price it cannot see its own edit

Six slots, all `rc=0`, all five of the 110th audit's `FOR THE BUILDER` items
discharged, **no threshold moved in any direction**, `run verify` re-judges 110
PASS entries with **0 verdicts that no longer re-derive and 0 gates that ignore
their control**, all 110 PASS commits still resolve in git, and **0 PASS rows
carry a `dirty_files` stamp**. Section 2 has no silent loosening to report and I
say so plainly.

What I found instead is one module reproducing its own founding scar, in the
exact direction it was written to catch, two days after it shipped.
`experiments/stale_cost.py` exists to answer *"what does THIS edit cost the
scoreboard, priced BEFORE the commit"* — and it computes "already stale" by
hashing the file **from disk**. Once the edit is in the working tree, a
certificate staled **by** that edit is indistinguishable from one that was stale
**before** it, so it is filed under `already` with the words *"a debt, but not
this edit's bill"* and the bill prints **0**. That happened at 19:16 yesterday to
`T0.28` — the honesty fixture for `decisions.py`, the instrument behind this
organ's own SECOND mandated check — and its PASS has been standing on older code
for five slots since.

---

## RANK 1 — `run stale-cost` CANNOT CHARGE THE EDIT IN FRONT OF IT, AND `T0.28`'s PASS IS THE UNPAID BILL

`T0.28` ("the escalation tool can be shown catching a deadlock and a
claim-death") is the ledger's certificate that `experiments/decisions.py` —
*"the instrument that stands over eleven pre-registered constitutional
defaults"* — detects every defect it claims to detect, in both directions, over
19 properties. It is the gate under this organ's SECOND mandated check.

Its standing PASS is **stale**, and it has been since `eca5757` at 2026-09-23
19:16.

### The proof, reconstructed rather than argued

`T0.28` declares `IMPL_DEPS = ["experiments/decisions.py"]`. I rebuilt its
`impl_sha` from committed blobs through the project's own one true code path
(`protocol.impl_sha_of`, with `file_bytes`/`dep_bytes` at each tree state):

| tree state | `impl_sha` of `T0.28` | decisions.py bytes |
|---|---|---|
| `1f32522` (the commit the row records) | `cffb8b1b90d5546d` | 114,661 |
| **`eca5757^`** | **`cffb8b1b90d5546d`** — exact match to the recorded row | 114,661 |
| **`eca5757`** | **`3543b4eebff2abe7`** | 115,269 |
| `HEAD` | `3543b4eebff2abe7` | 115,269 |

The recorded row (`ran_at` 2026-09-18T22:17:25, `dirty_files: null`) matched the
tree **exactly** up to and including `eca5757^`. The hash moved **at**
`eca5757`, and nothing else in the interval touched either file (`git log` on
both paths since 09-18 returns `1f32522` at 22:16:20 — one minute *before* the
run — and `eca5757`, nothing else). **`eca5757` and nothing else staled `T0.28`.**

### What the commit that staled it said about itself

`eca5757`'s own message: *"T0.28's `_experiment`/`_check` replayed dry → True (no
ledger write; **its standing stale-ness predates this edit** per stale-cost: 'a
debt, but not this edit's bill')."*

That sentence is false, and it was the reason the re-buy was declined.

### The instrument agrees with the false sentence, and here is why

```
$ run stale-cost experiments/decisions.py
  STALE-COST — 1 changed path(s) priced … **0 standing PASS certificate(s) would be staled**
    already T0.28     PASS  CHANGED BEFORE this edit — a debt, but not this edit's bill
```

`price()` classifies a certificate as already-stale from
`staleness_of(entry, path)`, and `impl_sha_of` reads the file **and its declared
deps from disk**. The edit is on disk by the time anyone prices it, so the
certificate is *already* `CHANGED`, `CHANGED` is in `ALREADY_KINDS`, and the row
falls out of `bill` into `already`. The default rendering path makes this
unavoidable: `render()` calls `price(changed_paths())`, and `changed_paths()` is
`git diff --name-only HEAD` plus untracked — **by construction it can only ever
name paths that have already been edited.** In that lane the `BILLED` branch can
fire only for a row whose staleness kind is *outside* `ALREADY_KINDS`, i.e. only
for the two pre-`impl_sha` rows. **For every properly stamped certificate, the
default lane's bill is structurally zero.**

I confirmed this is an ordering trap and not the module being wrong in general —
priced against a path that is **not** yet edited, it bills correctly:

```
$ run stale-cost experiments/coverage.py     →  BILLED  T0.21  PASS  re-buy costs cpu<1min
$ run stale-cost experiments/run.py          →  BILLED  T0.36  PASS  re-buy costs cpu<1min
$ run stale-cost experiments/decisions.py    →  0 billed; already T0.28
```

`coverage.py` and `run.py` are clean on disk (both certificates were re-bought),
so their dependents price correctly. `decisions.py` carries its edit, so its
dependent prices to nothing. Same tool, same tree, three invocations.

### Why this is RANK 1 rather than a 65-second errand

**The module's own docstring names this exact failure as the scar it was built
from.** Verbatim: *"`run stale` answers 'which certificates are stale NOW'. That
is the same question one commit too late… It never charges twice… That
distinction is the one the scar got wrong in BOTH directions in a single slot:
`T0.21` was stale before the builder touched anything (and was correctly
re-bought), **`T0.36` was staled BY the builder (and was not)**."* The module
shipped 2026-09-22 to tell those two cases apart. On 2026-09-23 it put `T0.28`
in the wrong one, in the same direction, and its answer was quoted into the
commit message as authority.

**The corroboration is in the same window, and it is the builder's own.**
`3e9cf42`'s message says the `T0.36` bill was *"priced **before** the edit"* —
and that bill was paid one commit later (`92cb7bd`). `T0.21`'s bill was the
Review's and was found four slots late by reading `run status`'s STALE block, not
by `stale-cost`. So of three certificate-staling edits in this window, the two
that were paid were priced before the edit or found by a different reader, and
the one priced after the edit was mis-billed. **This is not carelessness. The
discipline is working; the price tag is wrong.**

**The certificate is the one that guards a mandated check.** The code that moved
is the `class: conduct` deadline comparison — the exact path that decides whether
an entry on the owner's desk can go red — and the certificate asserting that
`decisions.py` "detects every defect it claims to detect" now describes the code
as it stood before that path existed. `decisions --check` printed `D33 (due
2026-09-23, STALE by 1 day(s))` for me tonight, correctly, on the strength of an
unstamped change.

**Aggravating, and it belongs in the record.** `308e202` at 00:14 tonight
journalled *"The remaining STALE rows are all settled FAILs behind Review
redesigns — no re-run legal"* while `run status` listed `T0.28  recorded PASS`
three lines above the rows it was describing. That slot re-bought `T0.21` for
precisely this shape of defect and read past the other instance of it in the same
block. It is the same failure the builder wrote into its own memory an hour ago —
*grep the STALE block for PASS rows specifically* — and `T0.28` is the row that
lesson was about.

**Live exposure is exactly one certificate** (`run stale` lists `T0.28` as the
only PASS row in the block) and the re-buy is **65.3 s of CPU**. The material
cost is small. The mechanism will repeat on every instrument edit this project
makes, and this repo edits instruments every day.

---

## RANK 2 — FOUR `GOAL.md` CITATIONS WERE RE-PARENTED TO A DECISION THAT HAD CLOSED FOUR DAYS EARLIER, AND HAVE BEEN OWNED BY NOBODY FOR EIGHT DAYS

`coverage` prints, and has printed since 09-02:

```
4 NEW unrunnable citation(s) — fix GOAL.md's text or route the revival;
never add to GOAL_UNRUNNABLE_BASELINE (shrink-only): GEN.02, GEN.03, GEN.06, GEN.09
```

`goal_unrunnable = 7 (unchanged since 2026-09-05)`. Three of the seven
(`DP.02`, `DP.03`, `LC.04`) are inside `GOAL_UNRUNNABLE_BASELINE` and read as
known debt. These four are not, and they are the red in `coverage`'s exit code
that the 110th audit reported to the owner as a standing indictment. **What no
audit has reported is that nobody holds the repair.**

### The pair, printed by an instrument that defers the judgement to me

`run review-queue`, `DISPOSITION-ON-A-CLOSED-DECISION` (the 100th audit's B2 —
*"a terminal row is never re-read, so a parent that closed before — or without —
inheriting leaves the work owned by nobody"*), declared **"A READING, never a
violation and never floored: whether the closure honoured the re-parent is a
human's judgement over the printed pair"**:

```
reparenting-the-welded-fifteen             -> D10 (closed 2026-09-01)
lt01-c2-body-cannot-rise                   -> D8  (closed 2026-09-01)
goal-cites-four-specs-that-resolve-to-corpses -> D24 (closed 2026-09-12)
```

It has printed these three pairs since 09-18 and **no audit has adjudicated
any of them.** I grepped this file's predecessors for `D24` and for the class
name: nothing. So I adjudicate all three now, and **two of the three are benign**
— saying so is the point of an adjudication:

- **`reparenting-the-welded-fifteen` → `D10`: BENIGN.** The row's own `ANSWER`
  (09-16, `34116ca`) is *"NO RE-PARENT IS OWED"* — the three weld roots are
  VOID-on-a-run, the repair is a successor spec, every dependent's `depends_on`
  stays untouched. `D10` is cited, not inherited from. Nothing is owed to it.
- **`lt01-c2-body-cannot-rise` → `D8`: BENIGN.** ACTED 2026-09-19 in `4091066`
  with the work actually done — C2' implemented at `b16de57`, `LT.01` attempt 2
  ran to PASS, the 0.6 m bar unmoved. `D8` is cited as history.
- **`goal-cites-four-specs-that-resolve-to-corpses` → `D24`: REAL, and it is
  the one that matters.**

### The one that is real

That row was stamped **ACTED 2026-09-16** (`34116ca`). Its `ANSWER`, verbatim:

> **Group B — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`** … **THIS DESK DECLINES
> THEM AS A DESIGN QUESTION** … Group B is **re-parented to `D24`'s
> resolution**, not to a date. **Whoever closes `D24` inherits these four.**

`D24` was closed **2026-09-12 ~17:3x UTC** — `docs/DECISIONS_RESOLVED.md:939`,
**RESOLVED BY ARMED DEFAULT: (iii) DECLARE, DO NOT DECIDE. The Learning-core
seat's arena is `VENUE-UNAFFORDABLE`. The "~10x" scale ratio is UNTOUCHED.**

Three things follow and none of them is a matter of interpretation. **A closure
cannot inherit a re-parent written four days after it** — there was no such
clause to inherit on 09-12. **The closure it was handed to decided nothing that
could revive the four**: `LC.07` stays `PILOT-BLOCKED`, the arena stays
unaffordable, and `(iii)` is by name the option that refuses to rule. And the
row is **ACTED**, which is terminal, so by the 100th audit's own reasoning
nothing will ever re-read it.

**Net effect:** four ids that `GOAL.md` cites in the present tense resolve to
specs nobody may run, `coverage` has exited 2 on them for 22 days, and since
2026-09-16 the repair has had **no owner, no date, and no desk**. The Review's
refusal to write a fourth date was right — the row says so well (*"writing a
fourth date for a repair whose precondition is somebody else's open decision is
manufacturing a promise this desk cannot keep"*) — and its refusal to delete the
citations was right and is the `champions.py` prohibition applied correctly. The
defect is narrow: **the parent it chose was already a corpse, and stamping the
row ACTED closed the only door back.**

---

## RANK 3 — `D34` FALLS DUE TODAY ON A REALISED-COST CLAUSE THAT HAS BEEN FALSE FOR SIXTEEN HOURS, AND THE CORRECTION MAKES ITS DEFAULT *MORE* EXECUTABLE

`D34` (`decide_by` 2026-09-24, armed) prices itself: *"The cost is REALISED, not
forecast: 24 hours, 0 iterations, 0 ledger events, and 30 perishable GPU-hours
with a legal buyer and no organ able to spend them."*

Measured tonight: **16 consecutive `iteration end rc=0`** from 2026-09-23T09:22
to 2026-09-24T00:15, zero non-zero exits in the 24-hour window; the ladder went
109 → 110 (`T4.06` FIRST-EVER PASS 09-23T10:46); and the legal buyer **was
spent** — `T4.06` drew 0.4387 h on Kaggle and harvested at 11:09. Every clause
of that paragraph was true when written and none of it is true now.

This is the **fourth instance** of the shape the 109th audit named in `3028801`
(*"a premise dies under a deadline"*) and I cite it rather than re-lesson it. But
the direction matters and is the reason this is RANK 3 and not RANK 1: `D34`'s
armed default is *"(iii) BOTH, IN THAT ORDER"*, and limb (i) is conditioned on
**"once running … ONLY after verifying in the same slot that `claude -p` reads
stdin on this harness."** A running builder is precisely what limb (i) was
waiting for. **The premise's death unblocks the default rather than undermining
it.** The underlying defect is untouched and real: `ladder_prompt.md` is
**88,330 B**, 42,742 below the 131,072 `MAX_ARG_STRLEN` exec cliff, growing
**+1,238 B/day** over 28 commits — **35 days**. Past the cliff the builder does
not read a degraded prompt; it does not start, and the slot looks like an
ordinary `rc=126`.

---

## The audit, item by item

**1. Integrity of the ledger — clean apart from RANK 1.** `run verify` EXIT 0:
110 PASS entries re-judged from the record alone, 107 controls probed, **0
verdicts that no longer re-derive, 0 gates that IGNORE their control, 0 gates
that could not be replayed, 0 entries that could not be audited.** My own three
checks beyond the tool: **all 110 PASS `commit` fields resolve to live git
objects** (`git cat-file -e`, 0 failures); **0 PASS rows recorded with
`dirty_files`**; every PASS has an implementation in `experiments/tests/`.
Standing and unchanged: **2 PASSes with no control at all — `T0.01`, `T0.10`** —
existence claims whose gate was never shown capable of reporting the bad case.
One stale PASS: `T0.28` (RANK 1).

Observation, not a finding, because the repair is a design pick and not mine:
**`experiments/stale_cost.py` is in no spec's `IMPL_DEPS`** (only
`experiments/run.py` imports it). The module that prices every other
certificate's re-buy carries no certificate of its own; it is gated by its own
`_check()` selftest alone. `T0.36` covers the *wiring* (`run.py`), not the
pricing logic.

**2. Thresholds and controls over 7 days — NO SILENT LOOSENING. Two movements,
both declared, both justified by measurement, and neither buys a certificate.**
I read `git log -p --since="7 days ago"` over `registry.py`,
`registry_expansion.py` and `experiments/tests/` — 24 commits.

- **`8f7d1dc` (09-19, `PS.08`): gate floor 0.015 → 0.008.** A loosening in
  direction. Justified in the message with a measurement: *"the 0.015 and every
  annotation beside it were a superseded variant's readings, measured not
  reproducing"*, re-frozen **against the final fixture before anything was
  committed**, and the movement was disclosed as inherited from a `rc=124`
  timeout. **`PS.08` then FAILed anyway** (2026-09-19T17:32, FIRST-EVER). No
  certificate rests on the loosened floor. Accepted.
- **`da07ede` (09-19, `LT.02`): cost class `cpu<2h` → `cpu<10min`.** Loosens
  admission and **TIGHTENS** the child-kill window 54,000 s → 10,800 s, on a
  written SIZING RECORD (68 s pilot, ~12–15 min projection), with the message
  stating no ledger row existed under the old class. Accepted.
- Everything else moved the other way: **`acad758` strengthened `T2.06`** (a
  strict `>` was deciding at zero margin; CLAIM 2 gained an exogenous margin),
  **`181fbff` gave `T3.06`'s claim two more comparators, one expected to kill
  it**, **`875caf6` derived `T3.06`'s dwell cap from `n` instead of typing it**,
  **`955b9ef`** amended `HR.1`'s registry text to the venue actually delivered
  while leaving `LEAK_EXCESS` 0.05, `CONTROL_FLOOR_EXCESS` 0.15, 17 features and
  3 seeds untouched, and **`1f32522`** recorded an honest `T0.28` FAIL before
  adding property 19.
- No `_check` gained an `or`. No control was deleted or weakened. No seed count
  was reduced. No assertion was removed.

**3. Drift from the goal — no drift, and a fourth consecutive window with
nothing about Jack in it.** The six slots bought: `HR.1`'s registry amendment,
`decisions.py`'s conduct-stale comparison, `anchor_margin`'s reader plus the
`T0.36` re-buy it billed, `steering.py`'s false-positive silencing plus one
routed class finding, a ratchet reading, two journal lines, and the `T0.21`
re-buy. Every unit traces to GOAL.md's *"protects the honesty of watching what
happens when the three meet"*. **None of it builds the brain, the body or the
world**, and the PASS delta for the window is **0**.

That is the correct purchase on this board rather than drift, and the board says
so mechanically: `coverage`'s QUEUE DEPTH reads **4 dispatchable today, of which
4 VOID → 0 FRESH dispatches**, and **5 of 7 cost classes have NO PATH IN** —
nothing runnable to implement and nothing gate-provisional to pilot
(`cpu<1min`, `cpu<10min`, `cpu<48h`, `gpu<20min`, `gpu<8h`). The builder
re-derived this from the tools rather than inheriting it, four slots running, and
refused manufactured work each time. I endorse every one of those refusals.

The converse, which is the harder half: **`0` commitments with no declared spec
(at floor), `4` CLAIM-DEAD, `13` with live claim specs and nothing passing.**
The named neglect classes from GOAL.md, measured: **curiosity 12 specs / 2 PASS;
one brain-unison 28 specs / 1 PASS; hearing 14 specs / 1 PASS (and that PASS is a
sensor fixture); learning-by-living — `thermal (kills)`, `shelter/building`,
`smell`, `balance` all CLAIM-DEAD**, every claim spec behind them parked or
foreclosed on honest evidence. `NO-LIVE-PATH` reads **7 distinct
commitments/seats with no live path**, lower bound 6.

**4. Is the builder alive and productive? Yes — 16 for 16.** 24-hour window
2026-09-23T00:37 → now: **16 iteration ends, 16 `rc=0`, 0 failures, 0 pace-skips,
0 STOPPED slots.** PASS delta **109 → 110** (`T4.06`). In my own six-slot window:
6 `rc=0`, 2 ledger events, both honest re-buys, PASS delta 0. Meters read by the
builder each slot: `week:all models` 16–17% against no pacing line — the gate
this project acts on. **The 26-slot blackout the 109th and 110th audits reported
is over** and the dark-slot counter that missed it was repaired at `e0786a0`
(replay of 09-23T08:07 now reads the true 26 against the 0 it printed live).
`review_liveness` (the schedule half) reads **OK — 2026-09-23 daily, 2026-09-20
FULL**. No repeated identical failures, no paused loop, no credit exhaustion, no
aborts on load.

One thing working as designed and worth recording as a positive: the Review's
09-23 sitting **exited `rc=124` mid-report and `docs/PROGRESS.md` is sealed with
an INCOMPLETE-RUN banner** (`51efe92`, `f1eee76`). The builder read that banner
in each of the four slots since and treated `ladder_prompt.md` `1^12` — committed
before the timeout — as the binding copy. The seal is the 76th audit's B4 doing
exactly its job: the schedule half says the desk opened, the seal says the
sitting did not finish, and the two no longer contradict each other silently.

**5. Compute honesty — no waste this window, and the perishable clock is now two
days out.** `2026-W38` (Sunday-start, matching Kaggle's reset): **0.9176 h drawn
of 30.0, 29.0824 h remaining, 0 unattributable, 0 failed hours.** Those ~29 h
**expire at the Sunday reset, i.e. end of Saturday 2026-09-26 — two days.** The
single dependency-satisfied buyer this project produced was `T4.06`, spent for
0.4387 h, harvested, PASS. **No GPU job since 2026-09-23 11:09.** There is no
second legal buyer: `coverage`'s `gpu<20min` and `gpu<8h` are EMPTY with no path
in, and `gpu<2h`'s only entry is `UB.10` (VOID, an arm to repair). Neither the
builder nor I will manufacture a dispatch to spend expiring hours; that
judgement is now consistent across four audits. Standing and unchanged:
**48.42 GPU-hours recorded against specs holding no verdict, 33.78 of them
`D1.0`'s alone across two attempts**, all behind `T1.08`;
`gpu_unattributed_jobs` **21, at its declared floor**.

**6. Stuck decisions — `decisions --check` EXIT 0, nothing escalated, nothing to
arm.** Ratchet `0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished-owner-ask,
0/0 default-action-expired, 0/0 firing-diff`. **No `MEANS-ESCALATED`** — no fork
a measurement could settle is on the owner's desk. **No `UNDECLARED`**, so the
standing instruction to arm at least one per audit has nothing to act on; that is
the honest result, not an omission. Live: **`D33` `CONDUCT-DESK`, due 2026-09-23,
now printing `STALE by 1 day(s)`** — the 110th audit's item-3 repair firing on
its first real instance, and the entry is the Review's to execute and report, not
to ask about. **`D32` and `D34` fall due TODAY** (both armed; both fire 09-25 if
unanswered — `D34` is RANK 3). `D31` due 09-25. Three entries still read
`CONDUCT-MISFILED?` (`D31`, `D32`, `D34`); I leave them as the 101st audit did.
**Nothing was quietly acted on without being recorded:** `D29`'s firing is a
closed `DECISIONS_RESOLVED` entry with the caveat transcribed verbatim onto
`CHAMPIONS.md`'s Learning-core cell (`096a8ab`), and `champions --check` still
reads `UNVERIFIED VERDICTS 2/2` — I re-ran it, the `HELD: BY VERDICT` marking did
not move. **The converse is RANK 2:** one decision was closed *without* the
re-parent that named it.

**7. Bakeoff hygiene — one reading, no violation.** No bakeoff ran in this
window. `T4.06`'s certification is the item on the board and its number is now
**visible in a live path** rather than derivable — the 110th audit's FTB 2
discharged in `3e9cf42`/`92cb7bd`, and `run status` prints:

```
T4.06 (PASS, attempt 1)  min_modality_latent_r2 vs 'incumbent' at min:
  grad_norm        refuted   margin -0.1529 = -56.7% of anchor seed spread 0.2699; 0 improving / 3 REGRESSING
  loss_reweight    CERTIFIED margin +0.0187 =  +6.9% of anchor seed spread 0.2699; 2 improving / 1 REGRESSING
  modality_dropout recorded  margin -0.2869 = -106.3% of anchor seed spread 0.2699; 0 improving / 3 REGRESSING
```

The winning arm's conjunct (2) was decided at **6.9% of the incumbent's own seed
spread with one of three seeds regressing.** That is inside the noise and the
builder said so first, in the certificate itself (`f7900b5`: *"the ratio result
is the demonstrated thing, the latent-recovery conjunct is not to be quoted as
demonstrated"*) — which is why this is a reading and not a finding. **No decision
was made without a learning gate; no VOID was treated as a verdict; no winner
was chosen inside the noise margin and then quoted as if it were not.** The
adoption decision falls **tomorrow** (`t402-touch-drowns-audio-at-the-fusion-boundary`,
DUE 2026-09-25) and it is the Review's — see FOR THE REVIEW.

**8. The honest summary — are we closer to a curious humanoid, or only to a
longer list of green ticks?**

**Neither today. We are marginally closer to a scoreboard that cannot lie to
itself, and not one step closer to Jack.**

The ladder reads **110/254, 43.3%, and it has not moved since 10:46 yesterday.**
This window's two ledger events were re-buys of certificates *about this
project's own instruments* — `T0.36` (run.py) and `T0.21` (coverage.py). That is
honest work and it is owed work, and it is also the fourth consecutive window in
which nothing on the ledger was about a creature.

The bottleneck has not changed and it is not the builder. `run next` shows zero
fresh dispatches; five of seven cost classes have no path in; every claim-dead
commitment's repair is a **redesign** sitting on the Review's desk. That desk
holds **57 live rows**, disposed **5** against **15** arrivals over seven days,
has used **`DECLINED` zero times across all 76 routed rows**, and its own
instrument reports the drain as **UNBOUNDED**. Six rows went `OVERDUE` at
midnight; **13 live dated rows fall due today against a measured capacity of 6,
and 7 of them cannot be discharged.** The world edit that would let cold,
distance, mass and exertion charge Jack anything is **18 days old, designed,
unregistered**, and its row broke its fourth date at midnight.

So the answer to section 8 is the same answer as yesterday, and it is not the
builder's to fix: **a body with every sense is worth nothing in a world that
charges for none of them.** What I can report as genuinely better than yesterday
is narrow and real — a deadline class that could not go red now goes red, a
bakeoff margin that had to be recomputed by hand now prints, and a certificate
that was about older code has been found and priced. That is three fewer places
this project can deceive itself. It is not a rung.

---

## FOR THE BUILDER

1. **Re-buy `T0.28` — 65.3 s, foreground, from a clean tree. It is the only
   standing PASS in `run status`'s STALE block and `eca5757` is what staled it.**
   The proof is in RANK 1's table and you can re-derive it in one command:
   `impl_sha_of` over `eca5757^`'s blobs returns `cffb8b1b90d5546d`, which is the
   sha on the row. **A dry `_check` → True is not a ledger row.** If the re-buy
   **FAILs**, record the FAIL — the `supersedes_fail` lane did exactly that on
   09-18 and it was the right move. Adding a property that pins today's behaviour
   is legal; **moving a bar so the re-buy passes is not**, and nothing in the
   19 properties needs to move for this.
2. **`stale_cost.price()` must compute "already stale" against the PRE-edit tree
   state, not against disk.** This is the RANK 1 repair and it is the difference
   between the module's contract (*"priced BEFORE the commit"*) and its default
   invocation (`render()` → `price(changed_paths())`, which can only name paths
   already edited). **The machinery exists — do not build new hashing.**
   `protocol.impl_sha_of` already takes `file_bytes` and `dep_bytes` overrides
   for exactly this question, and `protocol.tree_reconstructing_sha` is the
   existing precedent for "the sha at a committed tree state". For each changed
   path take its `HEAD` blob (untracked → the `missing:` rule the function
   already implements), compute each candidate's staleness at **HEAD-minus-this-edit**,
   and bill the rows that were clean there and are stale now.
   **Keep the no-double-charge rule exactly as written** — it is correct, it is
   pinned by `stale_cost._check`, and once "already" means "stale before this
   edit" it finally means what its own render string says. **Reporting-only and
   unfloored as it is today: no cutoff, no exit-code change, no ratchet.** Price
   the bill of your own edit *before* you make it and name it in the commit, the
   way `3e9cf42` did.
3. **Extend `stale_cost._check` so the fixture would have caught this.** One new
   case is enough and it is the one the module's docstring already writes in
   prose: a PASS row, clean at `HEAD`, whose declared dep carries an uncommitted
   edit, must land in **`bill`** and not in `already`. Run the committed pre-fix
   code against it to prove it is load-bearing — the same demonstration you used
   for `steering.py` in `5b18cd3`, which was the right standard.
4. **Say the stale-cost bill out loud in every commit that edits an instrument,
   and say which lane you priced it in.** `3e9cf42` wrote *"priced before the
   edit"* and was right. `eca5757` wrote *"predates this edit"* and was wrong.
   The two sentences look identical to a reader and mean opposite things; name
   the ordering, not just the number.
5. **Do not touch `GOAL.md`, `GOAL_UNRUNNABLE_BASELINE`, or the four `GEN`
   citations.** RANK 2 is the Review's to re-own. Deleting a citation would clear
   `coverage`'s red in one edit and that is the `champions.py` prohibition
   verbatim.
6. **Still not yours to pre-empt:** `A4`, `T2.10`, `SO.07`, `SO.10`, `T1.08`'s
   pipeline repair, `UB.10`'s successor, the world-edit window, the `lc03` seat
   row, the `t306` venue row, the `W1.01`/`W1.03`/`W1.04` registration, `T4.06`'s
   adoption, `HR.1`'s family fate, and the `GEN` group's re-ownership. `BA.03`
   (c) stays refused — `run status` still flags `PROGRESS.md` item 4 as naming a
   spec the runner would refuse (`BA.03`, VOID-FORECLOSED), and that order should
   not be re-issued as written.

---

## FOR THE REVIEW (read at 06:37 — and read RANK 2 before you touch the docket)

**`GEN.02`/`GEN.03`/`GEN.06`/`GEN.09` have had no owner since you stamped
`goal-cites-four-specs-that-resolve-to-corpses` ACTED on 09-16.** You re-parented
them to *"`D24`'s resolution"* with the words *"whoever closes `D24` inherits
these four."* `D24` closed **2026-09-12**, four days earlier, by armed default
`(iii) DECLARE, DO NOT DECIDE` — so there was no closure left to inherit them,
and the one that happened decided nothing that could revive `LC.07`. **Your two
refusals on that row were both right** and I am not asking you to reverse either:
not a fourth date you cannot keep, not a deleted citation. What is owed is a
**live parent** — a new row that ages, or an honest `DECLINED` with the reason,
or a named entry on the owner's desk. Right now `coverage` has been rc=2 on four
ids for 22 days with nobody holding the repair, and the row that held it is
terminal and will never be re-read.

**`D33` is `STALE by 1 day(s)` and the instrument now says so out loud.** It is
`CONDUCT-DESK` — yours to execute and report, never to ask about. Your own
addendum (`d9f568f`) already makes ACT the cheap limb and prices it: registering
`W1.01`/`W1.03`/`W1.04` from design text published 2026-09-06 is *"desk-shaped,
small, and mine"*, at the cost of `unreachable` rising above its floor of 96 with
a growth-log entry. `w1-world-edit-window` is **OVERDUE** and five other rows
went red at the same midnight.

**Before you adopt `T4.06` tomorrow, read what the new reader prints.** The
winning arm cleared `min_modality_latent_r2` by **+0.0187 — 6.9% of the
incumbent's own seed spread, with one of three seeds regressing.** The builder
wrote that caveat onto the certificate itself and it is binding: **the ratio
result is the demonstrated thing; the latent-recovery conjunct is not to be
quoted as demonstrated.** An adoption that cites `loss_reweight CERTIFIED` as
evidence the worst modality's recovery improved would be quoting a number your
own design pre-registered against being read that way.

**And the seat nobody can unseat is still unmoved.** `champions_trigger_debt` **3**
since 09-03, `champions_unwinnable` **4** since 09-13 — 21 and 11 days. The World
seat is held **BY VERDICT** with **no `VERDICT:` and no `TRIGGER:` declared at
all**, which means the seat behind this project's largest standing result cannot
be contested by any evidence. It belongs to the same docket as `D33`.

---

## FOR THE OWNER

**1. NO-DECISION: one of your certificates has been standing on code that
changed, and the tool built two days ago to prevent exactly that agreed it was
fine.** `T0.28` is the ledger row that says your escalation instrument —
`decisions.py`, the thing that stops decisions rotting on your desk — detects
every defect it claims to detect. Yesterday at 19:16 your builder repaired that
instrument, correctly, as I ordered. That repair changed the file `T0.28`'s
certificate is *about*, which makes the certificate owed a 65-second re-run. The
module whose entire purpose is to print that bill printed **zero**, and labelled
the debt as somebody else's pre-existing one — because it decides "was this
already stale?" by reading the file on disk, and by then the edit is on the disk.
Its own docstring names this precise confusion as the scar it was built from.
**Nothing is unsound: the science was replayed and passed, no threshold moved,
and the exposure is one row and one minute of CPU.** It is here because the
failure mode is *a checker that cannot see the change it is being asked about*,
this repo edits its instruments every single day, and the repair is ordered as
the builder's items 2 and 3. **Nothing to rule.**

**2. NO-DECISION, and it is the one I would not want you to skim: four sentences
in `GOAL.md` are false in the present tense, and since 09-16 nobody has owned
making them true.** `GOAL.md` cites `GEN.02`, `GEN.03`, `GEN.06` and `GEN.09` —
the generality claims: mastering two worlds and abstracting "shelter" from
"lean-to", other minds, transfer — as things being tested. All four are welded
behind `LC.07`, whose venue your own `D24` declared **unaffordable** on 09-12 by
armed default, choosing *"declare, do not decide"*. Four days later the Review
declined them as a design question and re-parented them **to `D24`'s
resolution** — a decision that had already closed. So the work was handed to a
door that had shut, the row that handed it over was stamped terminal, and
`coverage` has exited 2 on it for 22 days with no desk holding the repair.
**I am deliberately NOT opening a new decision entry for this**, and I want the
reason on the record so a future audit can check it: `D24` already answered this
question once, by default, and manufacturing `D35` to re-ask what "declare, do
not decide" already refused would be re-litigating a closed ruling — which is the
failure this file exists to catch, not to commit. The mechanical repair is
re-ownership and it is the Review's; I have ordered it there. **What is yours, if
you ever want it, is narrower and is not urgent: whether `GOAL.md`'s generality
paragraph should keep citing four ids nobody may run, or say plainly that the
claims are real and the venue is not yet affordable.** Only you may edit that
page. A default may never do it.

**3. NO-DECISION: the perishable GPU clock, reported because `D30` requires it,
and it is now two days from zero.** `2026-W38` holds 30 free Kaggle
GPU-hours; **0.9176 drawn, ~29.08 h expire at the Sunday reset — end of Saturday
2026-09-26.** The one dependency-satisfied buyer this project has produced in
three weeks was spent yesterday (`T4.06`, 0.4387 h, PASS) and there is no second
one: two of the three GPU cost classes are empty with no path in, and the third
holds only a VOID. **Neither your builder nor I will invent a dispatch to spend
them.** That judgement is consistent across four audits now and I restate the
reasoning rather than the conclusion: a run whose result nothing may claim
converts free hours into a ledger row that has to be explained later, and 48.42
GPU-hours already sit against specs holding no verdict — **33.78 of them `D1.0`'s
alone**, behind `T1.08`.

**4. `D32` and `D34` fall due TODAY; `D31` tomorrow. `D34`'s cost paragraph is
now out of date in your favour.** `D34` (argv vs stdin) prices itself on *"24
hours, 0 iterations, 0 ledger events"* — measured tonight, the builder has run
**16 consecutive clean slots** and the blackout ended at the weekly meter reset.
That is the fourth time an entry on your desk has reached its deadline on a
premise that died underneath it, and it is cited rather than re-lessoned. **It
changes nothing about the default and slightly improves it:** limb (i) was
conditioned on *"once running … verify in the same slot that `claude -p` reads
stdin"*, and a running builder is exactly what it was waiting for. The defect
underneath is untouched and real — `ladder_prompt.md` is **88,330 B against a
131,072 B exec cliff, growing 1,238 B/day: 35 days**, and past the cliff the
builder does not read a shortened prompt, it does not start, and the slot looks
like an ordinary `rc=126`. **Recommendation unchanged and unchanged in wording
from the two audits before it: let `(iii)` fire.**

**5. The standing indictment, nineteenth day, and the only number here that
matters.** Four of your constitutional commitments have **no live falsifiable
claim at all** — smell, balance, shelter/building, and *too cold kills him* —
every claim spec behind them parked or foreclosed on honest evidence, and
`NO-LIVE-PATH` reads **7 distinct commitments or seats** with no way in. Your
builder ran 16 clean slots in 24 hours and could not touch any of them: **`run
next` shows zero fresh dispatches and five of seven cost classes have no path in
at all.** Every one of those holes traces to the same place — a world that
charges nothing for cold, distance, mass or exertion — and the world edit that
would change it **was designed 18 days ago, has never been registered, and broke
its fourth promised date at midnight.** The desk that owns it has disposed of 5
rows against 15 arrivals this week, has never once used `DECLINED` in 76 routed
rows, and its own instrument calls the drain **UNBOUNDED**. The builder is not
the constraint. **The design queue is**, and that is the sentence I would want you
to remember tomorrow.
