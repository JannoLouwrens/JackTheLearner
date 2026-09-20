# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **106th audit — 2026-09-20, 06:37–07:1x UTC.** Opened at HEAD `c779c4d`,
> 27 minutes after the 06:0x builder slot ended (06:10:38), **concurrently with
> the Review's Sunday FULL**, which landed five commits under me while I read —
> `b138c38` (06:42), `0a1b7f1` (06:44), `af21fe0`, `acad758` (06:46), `87c8f04`
> (06:47). Every number below was re-derived at `87c8f04`, after those commits,
> and one of them (RANK 3) is a ratchet that went red between my first reading
> and my last. The 06:37 collision is not a hazard to work around; today it is
> the only reason the red has a witness.
> `demonstrated` **109/253 (43.1%)**. Ledger: PASS 109, FAIL 30, VOID 14,
> BLOCKED 1. Builder, last 24 h (09-19 06:37 → 09-20 06:37): **24 slots
> started, 24 ran** — 23 × `rc=0`, 1 × `rc=124` (16:57, inherited and completed
> by the 17:0x slot). Zero verdict events in the last 9 hours; the board has
> been re-verified empty six consecutive times.

## VERDICT: DRIFTING — the week's only PASS is a verdict flip in which not one measured number changed, and the desk that ruled on it this morning reinterpreted its own binding stop-rule at the moment it fired

Sections 1, 4, 5 and 6 are clean and I re-derived every one. The builder's
conduct in this window is good and I will say so before anything else: 24 of 24
slots ran, five first-ever FAILs were harvested as found and routed the same
hour, the last constitutional coverage hole closed (`commitments_uncovered = 0`,
at floor), the lane guard learned to see the lane it names, and three days of
false journal record were reconstructed in the open rather than quietly
overwritten. None of what follows accuses anyone of hiding anything.

What is wrong is what the project's own headline numbers **mean**, in three
coupled places:

| # | finding | damage |
|---|---|---|
| 1 | **`LT.01`'s FAIL → PASS moved exactly one of 47 shared metric keys, and that key is `claim_branch` — the verdict's own text.** The sole conjunct that failed in attempt 1 was rewritten into a three-way branch recorder with no `return False` path | the ladder's headline spec, +1 on the week's only demonstrated delta, and nothing a reader of the ledger or `run status` meets says the verdict moved because the clause did |
| 2 | **A binding STOP-RULE fired this morning and the desk that wrote it substituted a different consequence.** `sh02-null-saturation` promised DECLINE on a fourth break; it was DISPOSITIONED at 06:42 and is **still OVERDUE**, still on a clock, now bound to `w1-world-edit-window`, itself 2 d overdue | `review_queue_violations` is unchanged at 15 after the FULL's bundled ruling; `0 DECLINED of 68 routed` in the file's whole history |
| 3 | **A shrink-only ratchet went RED at 06:47, during this sitting.** `unreachable` **96 → 98** of 253, grown by the Review's own (excellent) `T2.06` strengthening, which staled a certificate that blocks 3 specs. The growing commit raised no baseline and named no justification | the repair is routed and cheap (~20 GPU-min), but the ratchet's own rule was not paid in the commit that broke it |
| 4 | **One threshold DID move in the loosening direction in the window** — `PS.08`'s `GAP_ABS_MIN` 0.015 → 0.008 — where the 105th audit reported "zero numeric thresholds moved in the loosening direction across seven days" | the blanket sentence was false; the move itself was disclosed, pre-registration, and inert |

---

## RANK 1 — `LT.01` PASS: 47 shared metrics, 1 changed, and it is the verdict string

**The measurement, which is not in dispute anywhere.** `experiments/ledger.json`
carries `LT.01` at `attempt: 2`, `commit 414a1b1`, PASS, 2017.13 s, seeds 0/1/2,
with attempt 1's FAIL (`28a232e`, 2026-08-31) preserved in `history`. I diffed
the two rows key by key:

```
shared metric keys                47
  ... differing                    1   claim_branch
  ... identical                   46   incl. nonladder_rise_max 0.0843667,
                                       p_hang 0.0308333, platform_nonladder 0.0,
                                       burst_rise_ceiling 0.611167,
                                       hang_check_h 0.4847, z_rest 0.3896
control_metrics                    IDENTICAL, all five keys
new keys in attempt 2             13   all of them adv_* + c2_branch
```

This is by design and the design is honest about it: the adversary's rng stream
(`30_000 + seed`) is disjoint, so *"phases 1-4 reproduce attempt 1's numbers
exactly and the adversary is the only new physics"*
(`lt_01_null_floor.py`, pilot record). The 2017 s bought exactly one new
quantity.

**What failed in attempt 1, and what it became.** Attempt 1's `claim_branch`
reads *"no non-ladder route reached 0.6 m rise"* — the old C2, removed in this
window:

```python
-    C2 >= 0.6 m of non-ladder torso rise reached in at least one seed
-    if m["nonladder_rise_ge_bar"] <= 0.0:
-        m["claim_branch"] = "no non-ladder route reached 0.6 m rise: ..."
-        return False
```

Its replacement, `lt_01_null_floor.py:760-785`, is an `if / elif / else` over
`G`, `G-adv`, `U` that assigns `m["c2_branch"]` and falls through. **C2' has no
failing path.** The implementation says so plainly — *"its evidence being
incomplete is V5's VOID, not a FAIL"* — so this is disclosed, in the impl
docstring, where a ledger reader will not go.

Every other clause (C1 null floor 0, C3 hang band 0.0308 ∈ [0.01, 0.05], C4 no
alternate route, `oracle_platform` 0.0) was **already green in the FAIL row**. So
the PASS is carried entirely by conjuncts that were green on 2026-08-31, and the
verdict moved when the clause did.

**Is it legitimate? Yes, by this project's own law, and I am not asking for a
re-run.** The re-scope was ordered by a dated Review disposition
(`lt01-c2-body-cannot-rise`, 2026-09-06), the ground is sound — the old C2
demanded, as a precondition of a claim titled *un-gameable rise*, an observation
of gameability — the 0.6 m bar did not move in either branch, attempt 1's row
stays in history, a new VOID gate (V5) was **added**, and the adversary is real
adversarial work. The T1.02 precedent permits a redesign when the EXPERIMENT is
wrong, and this experiment was wrong in the way the disposition named.

**Three things are nevertheless not written down anywhere a reader will meet
them, and each is load-bearing:**

1. **The flip is definitional.** 1 of 47. Nothing in the row, in `run status`,
   or in `run show LT.01` distinguishes "a bar was cleared that had been missed"
   from "the clause that missed it stopped being able to miss".

2. **The recorded branch is 1.6 cm and one seed from its opposite.**
   `adv_rise_max` **0.615667 ± 0.0478151** against the 0.6 m bar — a margin of
   **0.0157 m, 0.33 σ** — and `adv_ge_bar` **0.667**, i.e. **2 of 3 seeds**; one
   seed read below. The branch rule is deliberately asymmetric: `G-adv` fires at
   `adv_ge_bar > 0.0` (one seed suffices), while `U` must be *"EARNED"* by the
   adversary failing everywhere. Under any symmetric rule — unanimity, or the
   mean less one σ — this row reads **Branch U**. The commit message does state
   "2 of 3 seeds over the unmoved 0.6 m bar", to the builder's credit. It does
   not state that the branch is the one a coin-flip margin defaults to. The
   pilot read **0.7142 m** at a third of the envelope and the registered run read
   0.6157: the quantity deciding this branch moved 0.10 m between two honest
   measurements of it.

3. **The conclusion recorded is not the conclusion the original clause would
   have supported.** `4091066` says *"the original C2 claim restored as a
   measurement"*. The original C2 was about **the null** — *"the same random
   agent reaches ≥ 0.6 m"* — and the null still reads **0.0844 ± 0.0667 m**,
   seven times under the bar. What cleared it is an arm with adhesion enabled
   and a 1 s lookahead optimiser. That may well be the *better* necessity
   argument for `h(t)` — the arms this ladder will actually run are optimisers
   with adhesion, not random walkers — but it is a different argument, and
   "restored" is the wrong verb for it.

**What this does and does not damage.** It damages no ledger claim: `LT.01` PASS
is defensible and I am not asking for it to be re-run or re-marked. It damages
the **reading**. The week's demonstrated count moved 108 → 109 exactly once, on
this row; `docs/PROGRESS.md` called the C2' work *"a genuinely good piece of
adversarial science"*; the 104th and 105th audits both recorded "LT.01 PASS
banked" as a win. All three are true. None of them says what the +1 is made of.
And the LESSONS.md rule this project wrote for itself after `PS.01` —
*"when a spec is revised, enumerate every clause and its DIRECTION — harder,
easier, unchanged — in the spec's own `notes` and in the commit message, and
name the easier ones first"* — was honoured in the implementation docstring and
**not** in `registry_expansion.py`'s `notes`, which describe C2' as a
"two-branch necessity test" without saying it can no longer fail.

The credit, stated as plainly as the finding: the unblock was real. `LT.01` PASS
freed `LT.02`, `LT.02` was implemented, piloted with its FAIL-side reading
disclosed in advance, run, and returned a first-ever FAIL that is a genuine
finding about the venue. That is the ladder working. It is also why the flip
matters: a definitional PASS propagated into real downstream science, and the
propagation is invisible.

---

## RANK 2 — a STOP-RULE fired at 06:42 and the desk that wrote it chose a different consequence; the row is still red

`sh02-null-saturation` has been live 21 days and has broken four dates
(09-06 → 09-09 → 09-13 → 09-19). On 2026-09-14 the Review wrote into the row, in
its own hand:

> **STOP-RULE, binding on this desk: if this date breaks too the row is
> DECLINED and the finding is carried to the owner as a class, because a
> promise renewed four times is not a promise and a row nobody will ever rule
> on should not be occupying a clock.**

The 09-19 date broke at midnight. At **06:42 today** the Sunday FULL ruled
(`b138c38`), and stamped:

```
| DISPOSITIONED 2026-09-20 (Review FULL — option (b) ... VENUE repair, so
  execution is bound to `w1-world-edit-window`. THE STOP-RULE FIRED AND IS
  DISCHARGED BY A RULING, NOT A DECLINE.)
```

**The substitution is defensible on the rule's stated ground and indefensible on
its stated consequence, and both halves are true.** The ground was *"a row
nobody will ever rule on"* — and the desk ruled, which is strictly more than a
decline would have bought. This was real design work on a real question and I am
not calling it evasion.

**But the mechanical effect is that nothing the stop-rule existed to change,
changed.** Measured against `experiments/run review-queue` at `0a1b7f1`, after
the FULL's commits:

```
sh02-null-saturation        DISPOSITIONED   still OVERDUE (promised 2026-09-19)
review_queue_violations     15              unchanged across the bundled ruling
DISPOSITIONED               12 -> 16        "NOT counted as disposal; the row is
                                             still live and still ageing"
disposed (ACTED/DECLINED)    9              unchanged, 1.29/cycle
DECLINED                     0 of 68 routed  — in this file's entire history
drain                        UNBOUNDED       51 live rows
```

The row carries **no new `DUE:`**. Its last date is the one that broke. So after
being ruled on it is still a live row, still occupying a clock, still red, and
its execution is now bound to `w1-world-edit-window` — which is itself **OVERDUE
by 2 days**. A row whose stop-rule said it should stop occupying a clock now
occupies a red clock behind a second red clock.

This repository already names this disease in another register: *"a deadline
that moves when it is reached is the deadlock it replaced"* (the `decide_by`
re-arm rule, quoted in my own standing brief). A **stop-rule** whose consequence
is reinterpreted by its author at the instant it fires is the same object. The
honest repairs are the three the queue itself lists — and one of them fits
exactly: **a new `DUE:` with the reason** ("execution bound to
`w1-world-edit-window`; this is a re-date, not a fourth renewal of the same
promise"). That costs the desk nothing it has not already paid and it takes the
row out of VIOLATION honestly. Marking it `HELD` would **not** be honest — the
queue's own rules make relabelling a live row `HELD` its own violation — and
neither would dropping the `DUE:`.

Two further facts, reported without a charge attached, because the pattern is
what matters: `ba03` (due today) and `t306` (due 09-21) were bundled into the
same ruling and each also carried a stop-rule; both are DISPOSITIONED, neither
is overdue yet, and both will be live rows tomorrow. And the FULL's entire
throughput this morning sits in the DISPOSITIONED column — **+4 designed, +0
disposed** — which is the desk's real product and is also, by its own reader's
definition, not a disposal.

**No instrument can see any of this.** `review_queue.py` reads `DUE:`,
`BLOCKED-BY:`, `ROUTED:` and the disposition token. A stop-rule is prose. This
one was written in the open, in the strongest language on the board, by the desk
it binds — and the only reason it is in this report is that I happened to read
the row's body. That is precisely the class of promise the `DUE:` reader was
built for in the first place (my own B4, 2026-08-31).

---

## RANK 3 — `unreachable` grew 96 → 98 at 06:47, inside this sitting, and the commit that grew it paid none of the ratchet's price

This finding is five minutes old at the time of writing and is the reason the
06:37 collision is worth running into rather than around.

At HEAD `c779c4d` (06:10) `coverage` read `UNREACHABLE ... 96 of 253 (38%),
baseline 96` — AT floor, as it has been since 09-19. At HEAD `87c8f04` (06:47) it
reads:

```
UNREACHABLE: 98 of 253 specs (39%), baseline 96, shrink-only.
!! unreachable specs GREW: 98 of 253 vs baseline 96. Growth is permitted only
   with a named justification in the commit that grows it — raise
   UNREACHABLE_BASELINE there, append to its growth log, and say WHY ...
   Otherwise the repair is an UNBLOCK.
```

**The cause, from `run blocked`:**

```
T2.06 = PASS but STALE — re-run it    frees 2 / blocks 3
T2.06=PASS but STALE + T2.07=FAIL     frees 1: T3.08
```

`acad758` (Review FULL, Part 2) **strengthened `T2.06`'s CLAIM 2** — a strict
`acc_lang > acc_tfidf_name`, which decides at zero margin, replaced by an
**exogenous** `MARGIN_LANG = 0.07` derived from the binomial alone
(`2·√(2·0.25/400) = 0.0707`) rather than from the observed spread, with the
registered run's per-seed margins (0.1050 / 0.1425 / 0.1250) disclosed *for
audit* rather than used to set the bar.

**The strengthening is the best single piece of work in this window and I want
that on the record before the finding.** It is a bar moving **up**, derived
before the data was consulted, disclosed with the numbers that would have
embarrassed it if it had been reverse-engineered, and it catches the
`hash-salt-lottery-in-a-gated-metric` class *in a second costume* — a
zero-margin `>` deciding a certified conjunct. That is Part 2 doing exactly what
Part 2 is for.

**The finding is the bookkeeping, not the science.** Strengthening `T2.06`
staled its certificate; a stale certificate is not a PASS for dependency
purposes; `T3.08` and two others fell out of reach; a **shrink-only ratchet
grew**. The tool names three legal responses and `acad758` took none of them: it
did not raise `UNREACHABLE_BASELINE`, did not append to the growth log, and did
not name the growth in its message. The number is red right now.

**In fairness, the repair is already routed and is cheap.** `87c8f04`, committed
six minutes later, corrects the steering page's own item 5 in-sitting —
*"W38 has a legal GPU buyer after all, and my own next act is what created
it"* — because the staled certificate is a `gpu<20min` dispatch whose dependency
`T1.01` is PASS. So the honest description is **transient growth with a named
repair**, not silent rot: re-buying `T2.06` under the new bar takes `unreachable`
straight back to 96, and the recorded margins say it should clear 0.07 with room.
Two things still have to happen for that to be true rather than assumed: the
re-run must actually land, and **if it FAILs under the new bar the growth becomes
permanent** and must then be paid with a baseline raise and a written reason. The
desk has already said the right thing about that too — *"MARGIN_LANG is not
negotiable against its own re-run's result."*

What I am recording is narrow: for the interval between `acad758` and the
re-buy, this project's `unreachable` ratchet is red, no commit explains it, and
the only reason it is explained anywhere is that an audit happened to be running
during the sitting that broke it. A ratchet that can be broken and repaired
between two readings is a ratchet whose growth log has a hole in it.

---

## RANK 4 — one threshold moved in the loosening direction, and the previous audit said none had

**§2 in full, re-derived rather than inherited.**
`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`. The window is dominated by
five new files (`ps_05`, `ps_06`, `ps_08`, `ps_09`, `lt_02`) and by the `LT.01`
C2' rewrite covered above. Scanning every removed line carrying a numeral:

- **`N_PROPERTIES` 15 → 16 → 17 → 18 → 19** across the T0.2x/T0.3x
  self-check batteries. Strengthening, five times.
- **`LT.01` C2 → C2'.** RANK 1. The 0.6 m bar is byte-identical in both
  branches; V5 is a **new** VOID gate; the clause's ability to FAIL was removed.
  Direction: the claim got easier; the aliveness proof got harder.
- **`LT.02` cost class `cpu<2h` → `cpu<10min`** (`da07ede`). Loosens admission,
  **tightens** the child-kill window 54,000 s → 10,800 s, on a declared SIZING
  RECORD (68 s pilot → ~12-15 min projection). No ledger row existed under the
  old class. Clean.
- **`T2.06` `MARGIN_LANG` added at 0.07, replacing a strict `>`** (`acad758`,
  landed 06:46 during this sitting). A bar moving **up** from zero, exogenously
  derived. Strengthening, and the best of the window — see RANK 3.
- **`PS.08` `GAP_ABS_MIN` 0.015 → 0.008** (`8f7d1dc`). **This is a numeric
  threshold moved in the loosening direction** — the world-gate floor on the
  light-vs-heavy cost gap, lowered from **above** the pilot's measured 0.0119 to
  **below** it, at 1.49×.

On the `PS.08` move, the complete and fair account:

- It was **disclosed** in the commit message, first line, with the reason:
  *"the 0.015 and every annotation beside it were a superseded variant's
  readings, measured not reproducing"* — the 16:0x slot timed out at `rc=124`
  and the inherited annotations belonged to a fixture that no longer existed.
- It was **pre-registration**, not post-hoc: the floor was frozen at 17:23
  (`8f7d1dc`), the registered run landed at 17:33 (`7502c96`). Nothing moved
  after a registered number was read, and the pilot seed (90) is disjoint from
  the recorded seeds.
- The composite gate is `max(GAP_ABS_MIN, QUANTUM_MULT × quantum)`, so the
  absolute floor is only ever one of two bars; the quantum bar was untouched.
- It was **inert**. The registered run read `gap` **0.0217**, which clears the
  old 0.015 as comfortably as the new 0.008, and `PS.08` FAILed anyway — on the
  probe (0.583 vs 0.70) with the amputated control out-reading it at 0.708. The
  loosening bought nothing and hid nothing.

So: legal, disclosed, justified by a measurement, and of no consequence. I
report it because the 105th audit wrote *"Zero numeric thresholds moved in the
loosening direction across seven days — stated as plainly as a finding would
be"*, and that sentence was **not true** on 09-19. A blanket clean bill in §2 is
the single most valuable thing this organ issues; it is worth exactly as much as
its accuracy, and an inert exception that goes unnamed is how the next one
becomes invisible. **Corrected reading: one threshold moved down in seven days;
it was pre-registered, disclosed with its measurement, and inert.**

---

## The rest of the audit, section by section

**§1 — integrity of the ledger. CLEAN, mechanically swept.** All **109** PASS
rows: every `commit` resolves in git (`git cat-file -e`, 0 missing); every spec
has an implementation in `experiments/tests/`; every spec declares a `control`;
**107 of 109** pass a `control_fn` into `run_spec`. The two that do not —
`T0.01` (repo imports clean) and `T0.10` (Kaggle round-trip) — declare
`control: NONE, BY DECISION (52nd audit B5)` with the reasoning recorded on the
spec. Correct, and checked rather than assumed. No PASS carries a `+dirty`
stamp. `LT.01`'s row is sound as a record; RANK 1 is about what it means, not
whether it is true.

**§3 — drift from the goal. No drift in what was built; the *ratio* is the
finding, and it is not new.** 21 commits in the window. By what they serve:
`LT.02`'s implementation, pilot, sizing record, run and routing (5) trace to
GOAL.md's curiosity section and to `LT.01`'s own family — this is the ladder
itself. The rest — the lane self-declaration, the 22-case fixture through the
real launcher, four instrument re-buys, the journal reconstructions, the two
LESSONS occurrence records, six slot journals — serve *"protects the honesty of
watching what happens"*, which GOAL.md's first principle explicitly admits as
in-scope. **Nothing in this window serves no GOAL.md sentence.** But ~5 of 21
commits touched Jack and ~16 touched the apparatus, which is the ratio
`PROGRESS.md` has flagged for a week and which today's six consecutive
verify-and-end slots make starker: from 22:13 on 09-19 to 06:37 today, **eight
and a half hours, zero verdict events, and the correct action in every one of
those slots was nothing**, because every path off the board runs through a
design answer the Review owes.

*The converse, which is the harder half.* `coverage` exits 2 with
`commitments_uncovered = 0` — **no commitment has zero declared specs**, the
first time that has been true, and the four `GOAL.md:187` primitives registered
in the last two days (`PS.05` far, `PS.06` tiring, `PS.08` heavy, `PS.09`
worth-it) are why. But **13 commitments have live claim specs and nothing
passing**, and **4 are CLAIM-DEAD** (smell, balance, thermal, shelter — every
claim spec parked or foreclosed, at floor since 09-19). The three GOAL.md claims
my brief names as most likely to be quietly neglected stand as: **curiosity**
2 pass of 12 specs, **one brain / unison** 1 pass of 27, **learning-by-living**
— `death & retry` 0 pass of 6, `hunger/thirst` 0 of 6, `sleep` 0 of 5. Those
five numbers have not moved this week.

`goal_unrunnable = 7`, unchanged since 09-05 — **the 104th audit's RANK 1, now
unmoved at 4½ days.** GOAL.md's present tense cites `GEN.02`, `GEN.03`,
`GEN.06`, `GEN.09` (all `welded<-LC.07`) plus `DP.02`, `DP.03`, `LC.04`: seven
ids that resolve to corpses. Its routing row,
`goal-cites-four-specs-that-resolve-to-corpses`, is stamped **ACTED** and
re-parents to **`D24`, which closed on 2026-09-12** — one of the three pairs
`review_queue`'s own `DISPOSITION-ON-A-CLOSED-DECISION` reading prints. A
constitutional red owned by nobody, behind a green `ACTED`. I am re-stating it
rather than re-finding it: it was correctly found, correctly reported, and has
not moved.

**§4 — is the builder alive and productive? ALIVE. Productive in the only way
open to it.** 24 slots started in 24 hours, 24 ran, 23 × `rc=0`, one `rc=124` at
16:57 whose unit (`PS.08`) the 17:0x slot inherited and completed. No PACING, no
ABORT, `lost_iterations.log` 0 bytes, no stray pids, `/data` 79 G free, HEAD
pushed, tree clean at every slot boundary I checked. PASS delta over 7 days:
**108 → 109 (+1)**; FAIL **25 → 30 (+5)**, every one a first-ever verdict
harvested as found and routed within the hour. That is the honest shape of the
week: one green, five falsifications, and the green is RANK 1. Six consecutive
verify-and-end slots is **correct behaviour** on an empty board and I am not
flagging it as idleness — `coverage` confirms **5 of 7 cost classes are EMPTY
with no path in**: nothing to implement, nothing to pilot.

**§5 — compute honesty. NO WASTE, and that is the problem — though it stopped
being strictly true at 06:46.** `overruns: []`. Zero GPU hours spent in the
window, zero in `2026-W38`, and the refusal was correct for every one of the
six slots that made it: every GPU-class pilot was behind a Review design row and
manufacturing a buyer is forbidden. **That changed mid-sitting**: `acad758`'s
`T2.06` strengthening staled a `gpu<20min` certificate whose dependency `T1.01`
is PASS, and `87c8f04` re-wrote the steering page's item 5 within six minutes to
say so. So `W38` now has **one** legal buyer worth ~20 GPU-minutes against 30
hours — the desk's own words, *"the inventory [is] overwhelmingly unbought"* —
and the large unblock still runs through `T1.08`'s undesigned pipeline repair.
The cost so far, from the tracker's own `weeks` key:

```
2026-W35   kaggle 18.9304 of 30      ~11.07 h expired
2026-W36   kaggle 17.7238 of 30      ~12.28 h expired
2026-W37   kaggle  1.3790 of 30      ~28.62 h expired
                                     ~51.97 h expired in three weeks
2026-W38   opened 2026-09-20, 30 h, expires Sat 2026-09-26
           legal buyers as of 06:47: ONE, ~20 GPU-min (T2.06 re-buy)
```

Against that, `gpu_hours_no_verdict` **48.42 h TOTAL** (33.78 h of it on `D1.0`
across 2 attempts and 0 verdicts). The project has now thrown away more free GPU
time in three weeks than the total it has ever spent without getting a verdict
for it. Neither number is the builder's fault and both are the same fact seen
twice: the ladder has no runnable GPU work.

**§6 — stuck decisions. CLEAN, and verified in code rather than inherited.**
`decisions --check` exits 0: **0 UNDECLARED**, 0 MEANS-ESCALATED, 0 OVERDUE,
`ratchet ok (0/10, 0/3, 0/0, 0/0, 0/0)`. Nothing is owed an arming this audit and
nothing fires from this desk today. **`D27` reads `due 2026-09-20`, which is
today, and is NOT overdue** — I re-derived this from
`experiments/decisions.py:1371`, which sets `overdue = (today - due).days`, and
the module's own note at line 232: *"marks a row overdue at
`(today - decide_by).days > 0`, so the earliest day a default can fire is
`decide_by + 1`"*. Earliest legal firing is **2026-09-21**, at the first slot.
Six journal entries and two audits asserted this; it is correct. Armed and
pending: `D27` (09-20), `D28` (09-21), `D29` (09-22), `D32` (09-24), `D31`
(09-25), plus my own predecessor's **`D28` RECLASSIFICATION NOTICE, which also
fires 2026-09-21**. Nothing was acted on without being recorded: `D20` and `D30`
both fired on 09-19 with the required wording and both are in
`DECISIONS_RESOLVED.md`. The five `CONDUCT-MISFILED?` soft flags (all five armed
entries) are unchanged and remain the 101st audit's standing observation.

**§7 — bakeoff hygiene. One reading, no violation.** `DECISIONS_RESOLVED.md`
adds nothing in this window. No decision was made without a learning gate; no
VOID was treated as a verdict *in the resolved file*. The live exception is
already indicted and already the owner's: the **Learning core** seat is held
`BY VERDICT` off `LC.03`, which is a VOID — `champions --check`'s
`VERDICT-IS-A-VOID`, 2 of 2 unverified verdicts, at floor. The near-miss worth
naming under this heading is RANK 1's item 2: `LT.01`'s **branch** was resolved
inside the noise margin (0.33 σ, 2 of 3 seeds) under an asymmetric rule. A branch
is not a bakeoff and no seat moved, so this is not a §7 violation — but it is the
same failure mode, and it is the reason I ranked it first rather than filing it
as a footnote.

**The standing reds, all at floor, none moved, none re-litigated.**
`champions --check`: 10 violations, `ratchet ok` — 2/3 unfalsifiable (ASR,
Speaker ID: `NO-ARENA`), 4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger
debt. `unreachable` 96 of 253 (38%), at floor. `claim_dead` 4, at floor.
`fail_unowned` 0, at floor — with `D23`'s warning louder than ever: **28 of 30
settled FAILs are owned by nothing but a dated promise from a desk whose drain
is UNBOUNDED**.

---

## The honest summary — §8, answered directly

**We are not closer to a curious humanoid that climbs the ladder than we were
yesterday, and we are barely closer than we were a week ago — but the reason is
worth more than the verdict.**

The week's green ticks went up by one and that one is a rewrite. The week's
*knowledge* went up by five, and those five are real: the world prices distance,
exertion, mass and risk on every seed, and in all four cases the **probe** — the
thing that was supposed to show Jack can read the price — collapsed, twice
out-read by an amputated control. `LT.02` found that this body's self-generated
chaos is *reducible*, so the detector certifying the curiosity family has no
reachable true positive in its own venue. Those are five falsifications of things
this project believed, bought honestly in a day, and they are worth more than
five PASSes would have been.

And then the ladder stopped, because every one of those findings is a **design
question**, and design is the one resource this system has run out of. Six
consecutive slots this morning correctly did nothing. Five of seven cost classes
have no path in. Thirty free GPU-hours opened today with ~20 minutes of legal
work to spend them on — and that 20 minutes did not come from the ladder, it
came from a desk strengthening a bar at 06:46 — after 52 hours died the same way
in three weeks. Fifteen dated promises are broken, 51 rows are live, the drain is
UNBOUNDED, and the one desk that can clear them spent this morning's sitting
moving four rows into a column its own reader defines as *not a disposal* —
including one whose binding stop-rule it declined to honour as written.

That last clause needs its counterweight, because the same sitting also produced
RANK 3's strengthening, which is the single best piece of work in this window and
which found a zero-margin gate nobody else had looked at. The Review is not
idle and it is not evasive. It is **outnumbered**, and the two findings I have
against it this morning are both what being outnumbered looks like from the
outside: a promise reinterpreted rather than kept, and a ratchet broken in
passing by a repair that was worth making.

So the shape of the project today is: **a builder that will run anything it is
given and has nothing to run, in front of a design queue that grows 2.9 rows a
cycle and clears 1.3.** That is `D28`, it is on the owner's desk, its default
fires tomorrow, and I have nothing to add to it except today's price.

The thing I would not want lost in the arithmetic: the system's honesty held
under pressure this week. A builder optimising for green would have banked
`LT.01`'s PASS and said nothing about 2-of-3 seeds; it put the number in the
commit message. It would have quietly re-piloted `PS.08` rather than write
"gate floor re-frozen 0.015 → 0.008" in the first line. It would have let three
unjournalled slots stay unjournalled instead of reconstructing them and marking
the reconstructed cause *"not established"*. That is the culture working. My
three findings are all cases where the honesty was written in the right place
and read in the wrong one.

---

## FOR THE BUILDER

Ordered. Items 1 and 2 are both **reporting-only, unfloored, gate nothing, move
no threshold**. Neither may change a verdict, a bar, or a ratchet floor.

1. **`VERDICT-FLIPPED-BY-RESCOPE` — a reading, in `run status`.** For every PASS
   row carrying `supersedes_fail`, compute and print: the count of **shared
   metric keys whose values differ** between the superseded FAIL and the PASS,
   over the total shared, plus `impl_changed` and whether `spec_sha` moved.
   `LT.01` reads **1 of 47, and the differing key is `claim_branch`**. Print the
   differing key names when the count is small (≤ 3). A PASS where nothing
   measured changed is not necessarily wrong — `LT.01` is not wrong — but it is a
   materially different object from a PASS that cleared a bar it had missed, and
   today no reader can tell them apart. Unfloored: some flips are legitimate and
   a gate here would forbid a legal move. Report the pair; the judgement is a
   human's. (This is the `DISPOSITION-ON-A-CLOSED-DECISION` idiom, not a new
   authority.)

2. **`STOP-RULE-FIRED` — a reading, in `run review-queue`.** A live row whose
   body contains the literal token `STOP-RULE` and whose latest `DUE:` has
   passed is printed under a `STOP-RULE` heading with the row id, the date that
   passed, and the current disposition token. Mechanical and cheap — no prose
   parsing beyond the literal token and the existing `DUE:` reader. **Never a
   violation**: whether the desk honoured its own rule is a human's judgement
   over the printed pair; whether anyone can *see* the pair is an instrument's.
   Today's live positive is `sh02-null-saturation` (`STOP-RULE`, DUE 2026-09-19
   passed, now `DISPOSITIONED`, still OVERDUE); `ba03-null-saturates-the-horizon`
   and `t306-matched-magnitude-noise-buys-coverage` are the same shape with
   dates not yet passed, so they are the known negatives to pin.

3. **Re-buy `T2.06` — it is the one legal GPU dispatch on the board and it is
   holding a red ratchet open.** `acad758` strengthened CLAIM 2
   (`MARGIN_LANG = 0.07`, exogenous) and staled the certificate;
   `unreachable` grew **96 → 98** because a stale `T2.06` blocks 3 specs
   (`T3.08` among them). `gpu<20min`, dependency `T1.01` is PASS, ~20 GPU-min
   against `W38`'s 30 free hours. The recorded per-seed margins
   (0.1050 / 0.1425 / 0.1250) say it should clear 0.07, **which is a forecast and
   not a licence**: harvest whatever comes back as found. **`MARGIN_LANG` does
   not move against its own re-run's result** — the desk said so itself. If the
   re-run FAILs, the `unreachable` growth becomes permanent and must be paid in
   that commit: raise `UNREACHABLE_BASELINE`, append to the growth log, name the
   reason. Do not raise the baseline before the re-run.

4. **One-line hygiene: `acc_tfidf_name` is an empty untracked file in the repo
   root**, created 06:47, almost certainly a stray shell redirect from the
   Part 2 re-examination of `T2.06`. Zero bytes, harmless, and exactly the kind
   of residue the slot hygiene check exists to catch. I left it rather than
   delete another organ's working-tree state mid-sitting.

5. **Enumerate the direction of `LT.01`'s C2 → C2' in
   `registry_expansion.py`'s `notes`.** One sentence, no code, no bar: the notes
   currently say "two-branch necessity test" and do not say that **C2' has no
   failing path** while V5 is a **new** VOID gate. The impl docstring says both;
   the registry is what `run show` prints. This is the `PS.01` lesson's own rule
   — *enumerate every clause and its DIRECTION, name the easier ones first* —
   applied to the spec it was written for. **No threshold moves, no re-run, no
   ledger edit.** `T0.36`/`T0.21` re-buys as usual if your edits stale them.

6. **Do not pre-empt the Review.** `sh02`'s missing `DUE:`, the `w1-world-edit-window`
   design, `A4`, `T2.10`, `SO.07`, `SO.10`, `T1.08`, `HR.1`, `UB.10` and the four
   `PS.*` legibility repairs bundled onto 09-24 are all that desk's. Naming
   candidate arms remains welcome; choosing one is not.

7. **Tomorrow, 2026-09-21, in order and not folded together:** `D27`'s default
   fires at the first slot (with the required wording — *"the owner did not rule
   by 2026-09-20, so the pre-registered default fired"* — plus how to reverse
   it), then `D28`'s, then the overseer's `D28` reclassification notice, then the
   two 09-21 dispositions (`WAITS-ON:` with `none` permitted; the `(iv)`
   **measurement** before the `(iv)` implementation).

---

## FOR THE OWNER

**1. NO-DECISION — the week's one green tick, reported because you read the
velocity number and it moved.** `demonstrated` went 108 → 109 this week. That +1
is `LT.01`, and in it **one of 47 shared measured quantities changed between the
FAIL and the PASS: the verdict's own text.** The re-scope that did it was
legitimate, dated, ordered by the Review on 2026-09-06, and disclosed in the
implementation — I am not asking you to reverse anything and the row must not be
re-run. I am telling you because the sentence *"the ladder gained a PASS this
week"* and the sentence *"the ladder measured something new this week"* are both
being said about the same event, and only the first is true of it. What was
genuinely measured this week is five **falsifications** — distance, exertion,
mass and risk are all priced by the world, and Jack's probe cannot read any of
them; and his self-generated chaos is reducible, so the curiosity family's
detector has no true positive in its own venue. Those are the week's science.
They are worth more than the tick, and they all landed on the Review's desk.

**2. NO-DECISION — a conduct fact about a desk, reported under your own
2026-09-17 ruling that the desks amend their own conduct, so there is no fork
here for you.** The Review wrote itself a binding stop-rule on
`sh02-null-saturation` — *"if this date breaks too the row is DECLINED"* — the
date broke, and at 06:42 this morning the desk marked the row `DISPOSITIONED`
instead, writing *"DISCHARGED BY A RULING, NOT A DECLINE"*. The ruling is real
work and the rule's stated ground (*"a row nobody will ever rule on"*) is
arguably met. The measurable effect is that the row is **still OVERDUE, still on
a clock, carries no new date, and its execution is now bound to a second row
that is itself 2 days overdue** — and `review_queue_violations` is unchanged at
15 across the whole bundled ruling. The repair is one line the desk can write
itself (a new `DUE:` with the reason), which is why this is a report and not an
entry on your desk. It is here because in 68 routed rows this queue has recorded
**0 DECLINED**, and today was the day the option was supposed to be exercised.

**3. `D28` — cited, not re-asked (`decide_by` 2026-09-21, fires tomorrow), with
today's price attached.** `D28` measures the Review's drain as UNBOUNDED. Today
adds three numbers it did not have: the queue took its Sunday FULL and came out
at **51 live rows, 15 still OVERDUE, 0 rows disposed** (+4 designed, which its
own reader defines as not a disposal); `2026-W38` opened this morning with **30
free GPU-hours and, until 06:46, no legal buyer at all**, after three weeks in
which **~52 free hours expired unspent**; and **5 of 7 cost classes have no path
in**. The builder ran 24 of 24 slots and correctly did nothing in six of them.
The one buyer that now exists is ~20 GPU-minutes and it was created by the
Review strengthening a bar, not by the ladder advancing — which is the cleanest
statement of `D28` available: **this project's compute is currently gated on one
desk's writing speed.** The backlog, the idle quota and the empty board are one
fact billed to three accounts.
Recommendation unchanged, and my predecessor's reclassification notice fires
tomorrow alongside the default.

**4. `D27` — cited, not re-asked. Its `decide_by` is TODAY, 2026-09-20, and it
is NOT yet overdue.** Verified in code, not inherited: `decisions.py` marks a row
overdue at `(today − decide_by).days > 0`, so the earliest legal firing is
**2026-09-21**, at the builder's first slot. If you want to rule on whether this
project buys mechanical certificate screens, today is the last day it is yours.
The default that will otherwise fire is `(i) BUILD THE SCREEN, REPORTING ONLY`,
unfloored, reversible by deleting one function from `experiments/coverage.py`.

**5. NO-DECISION — the constitutional red that has now not moved for 4½ days.**
`goal_unrunnable = 7`: `GOAL.md` cites `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`,
`DP.02`, `DP.03` and `LC.04` in the present tense, and every one resolves to a
parked, foreclosed or welded spec. The routing row that owned this is stamped
**ACTED** and re-parents to `D24`, **which closed on 2026-09-12** — so the work
is owned by nobody behind a green mark. This was the 104th audit's RANK 1, it was
found correctly, and it is unchanged. I am not re-routing it and I am not editing
`GOAL.md`; I am recording that it has now survived two audits and a Sunday FULL.
