# OVERSIGHT — 79th audit, 2026-09-06 18:37–19:0x UTC (at `12ae7ab`, tree clean)

## VERDICT: ON TRACK — **the ladder's integrity held under every mechanical check I ran, and the day's most important measurement contradicts the day's most important decision.**

The 78th audit's INTEGRITY RISK is **discharged**: `T0.35` is registered and
passing, `undeclared_impl_imports` walks every import node at full AST depth,
the grandfathered set is a named dict that shrank 35 → 27 → 9 in three commits,
and the eight `EpisodicMemory` importers now declare their subject. The
staleness lane can see its own blind spot. That was the right repair and it was
taken within six hours.

Section 2 is clean. I chased the single highest-risk event of the week — `ME.1`
going FAIL → PASS in eight hours on a control that had just been strengthened —
all the way to the diff, and **it is a genuine repair, not a loosening** (§2.1).
I found no threshold moved in the loosening direction, no control deleted or
weakened, no `_check` gaining an `or`, no seed count reduced, and no assertion
removed anywhere in seven days.

The finding that matters is not an integrity failure. It is that **`W1.00`
FAILed at 10:30 this morning measuring the exact hypothesis the Review had
published at 07:1x as its answer to a thirteen-day question, and nothing has
carried the result back** (§3.1). The ledger is honest. The reasoning built on
top of it is four hours out of date and no instrument in this repo will say so.

Ranked by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger — CLEAN

Mechanical sweep over all **106 PASS** rows of **245** registered:

- **0** PASS rows whose recorded `commit` fails `git cat-file -e <sha>^{commit}`.
- **0** PASS rows with no spec in `BY_ID`; **0** with no implementation module.
- **0** PASS rows whose spec declares no `control`.
- **104 / 106** carry populated `control_metrics`. The two that do not are
  `T0.01` and `T0.10`, and both declare `"NONE, BY DECISION (52nd audit B5)"`
  with the reasoning recorded in the spec — an import either raises or it does
  not; a sabotaged upload fails on the service's side. Those are argued
  exemptions, not silence. **No PASS in this ledger is a claim without evidence.**

Ratchet board, all at or inside floor: `unreachable` 94 vs floor 94, `fail_unowned`
0 vs floor 0, `review_queue_violations` 0, `champions --check` EXIT 0,
`decisions --check` EXIT 0, `run review-queue` 0 violations.

`coverage` exits 2, and that is the standing constitutional red (§3.3), not a
new break.

### 1.1 One mechanism worth naming: a dying organ committed a ledger it never finished reading

`review.log`, 2026-09-06T07:17:11:

```
sealed docs/PROGRESS.md as an INCOMPLETE RUN draft (rc=124)
committed 3 other dirty file(s) from the dying run:
  docs/CHAMPIONS.md, experiments/cpu_budget.json, experiments/ledger.json
```

The Sunday FULL hit its 40-minute wall and `lib_seal.sh` committed whatever was
on disk. One of those files was `experiments/ledger.json`, carrying `ME.1`'s
FAIL row (`24e4680+dirty`). **No damage here** — the row was a true measurement,
it was committed as found, and a clean-tree re-run superseded it at 14:19. But
the mechanism is general: an organ that timed out mid-reasoning commits the
ledger unbannered, and `PROGRESS.md` gets a banner while the ledger does not.
The seal already knows which files it is committing; it should stamp the rows it
writes the same way `+dirty` stamps a dirty tree. Named in FOR THE BUILDER.

---

## 2. Thresholds and controls — NO LOOSENING FOUND

### 2.1 The `ME.1` flip is a real repair, verified four ways

This is the one that had to be checked, so here is the whole chain.

| | Review run 06:51 | Builder run 14:19 |
|---|---|---|
| status | **FAIL** | **PASS** a8 |
| `spec_sha` | `0bef0b2f1432595f` | `0bef0b2f1432595f` — **identical** |
| `impl_sha` | `9908307cc8025299` | `a721df40d4d6011e` |
| `distractor_abstention` | **0.0000** ± 0.0 | **1.0000** ± 0.0 |
| `distractor_evaluated` | 40.0 ± 4.55 | 40.0 ± 4.55 — **identical** |
| `distractor_excluded` | 20.0 ± 4.55 | 20.0 ± 4.55 — **identical** |
| `cued_recall` | 0.8500 ± 0.0136 | 0.8500 ± 0.0136 — **identical** |

1. **The spec did not move.** `spec_sha` is byte-identical across the FAIL and
   the PASS. The 0.95 bar stands and the exclusion filter stands — precisely
   what the Review's FTB item 2 forbade touching.
2. **The test file gained nothing but a declaration.** `80f8c80`'s entire diff
   to `me_1_event_log.py` is `IMPL_DEPS = ["EpisodicMemory.py"]` plus a comment.
3. **The flip came from the implementation**, `6502d36` at 12:16 — similarity
   recalibrated to *coverage over the cue's known words*, floor 0.95, chosen by
   a 5-arm × 3-seed probe run through ME.1's own harness with every losing arm's
   numbers in the commit message (raw@0.34 abstention 0.000; raw@0.60 verbose
   recall 0.000 on all seeds; margin@0.20 terse 0.000 and recall down to
   0.63–0.70). The hole the winner opens is written into the docstring rather
   than hidden.
4. **The denominators are identical and recall did not move.** 40 evaluated / 20
   excluded in both runs — the control did not pass by evaluating fewer cues.
   `cued_recall` 0.85 ± 0.0136 matches the 08-30 and 09-02 rows to the digit.

**Abstention 0 → 1 at zero recall cost with no bar movement.** That is the
honest repair the Review asked for, delivered in six hours, and the cost half
landed as a committed `ME.3` FAIL rather than being absorbed.

### 2.2 The only assertion "removed" in seven days is a tightening

`git log -S` on the one deletion my sweep surfaced:

```
-        _worst(m, "sigma_life") >= SIGMA_MIN
+        min(_per_seed("sigma_life")) >= SIGMA_MIN
```

`LG.00` VOID attempt 1 — *"the certification gate fired at 19.94 and the true
worst seed is 21. The estimator was the fault, not the data."* The replacement
runs in the direction of distrust and the VOID row stayed. Correct.

### 2.3 One weakness, stated as a weakness and not a violation

`80f8c80` added four distractor conjuncts in one commit. Three carry
`MIN_DISTRACTOR_EVAL = 30`; **`ME.9` carries `MIN_DISTRACTOR_EVAL = 9`**
(`me_9_attributed_recall.py:116`). The denominator is 15 by construction (5
censored topics × 3 speakers), so 9 is 60% of it and the run reported 15/15 —
defensible. But it is the loosest aliveness floor in a family set in a single
commit, and an aliveness floor at 60% of a 15-item denominator tolerates a
control going two-thirds quiet. Worth a sentence in the file saying why 9.

---

## 3. Drift from the goal

### 3.1 **`W1.00` refuted the Review's Pile A diagnosis four hours after it was published, and nobody has carried it back** — the finding of this audit

At 07:1x the Sunday FULL dispositioned `w0-too-shallow` after thirteen days. Its
substantive claim was a two-pile split:

> **Pile A — UNDER-NULLED; the repair is in our instruments.** `LC.03`'s
> darkroom, `LC.03` v2, field watch wk5, `T3.06`. These findings are not wrong
> — they are *too kind to the learners*, because the honest null is harder.

`W1.00` was registered at ~09:3x to test exactly that and ran at **10:30:12**.
It is the right instrument, honestly built, and it says the opposite:

- **A stronger null does exist.** `repeat` scores `gain_repeat` **3.13** against
  white's **−0.019**. The first conjunct holds.
- **Re-scoring the eight recorded Pile A margins under it moves nothing.**
  Shift-over-own-std: `wk5_coverage` **0.022**, `ppo_needs` **0.029**,
  `ppo_lp` **0.036**, `darkroom` **0.037**, `dreamer_xs` **0.038**,
  `wm_efe` **0.038**, `wm_latent` **0.084**. Seven of eight, none within a
  factor of ten of the 1.0 gate.
- **The eighth is "cannot tell", not "moved".** `dwell` (T3.06) reads
  `shift_ratio` 16.2 but `fired_dwell` 0.0, because the spec's own guard
  correctly excludes it: the dw channel's noise floor `f_dw` **0.0082** exceeds
  that margin's own std **0.0062**, so the shift is unreadable. The guard is
  right and I checked it in the source (`w1_00_*.py:452-456`) rather than
  trusting the branch string.
- **Pre-registered branch fired:** `"immaterial: repeat outscores white but no
  recorded margin moves by more than its own std"`.

**So Pile A does not dissolve the shallowness findings.** It leaves seven
standing and returns "cannot tell" on the eighth. The world question therefore
tips *toward* Pile B — the world is genuinely shallow. This is the most
consequential number produced today and it currently exists as a bare `FAIL`
row with **no `REVIEW_QUEUE` row and no amendment to the disposition it bears
on.**

Note on where this question now lives, because it moved and the docs have not
caught up: **`D10` is RESOLVED** — the armed default fired 2026-09-01 and it is
off the owner's desk (`DECISIONS_RESOLVED.md:109`). The live homes for W0's
depth are the `w0-too-shallow` row (DISPOSITIONED this morning, on the reading
`W1.00` just contradicted), `w1-world-edit-window` (OPEN, DUE 2026-09-13), and
`W1.01`, which is the spec that would actually settle it and is not registered.

Nobody did anything wrong. The builder registered the spec the Review ordered,
ran it, and committed the FAIL as found — that is the process working. What is
missing is the step where a result invalidates the reasoning that ordered it.
`fail_unowned` reads 0 and will keep reading 0, because `W1.00` is attempt 1 and
unsettled; no instrument here watches for *"a fresh FAIL contradicts a
disposition made this week."*

### 3.2 `W1.00`'s control fired: our historical null-picking was fitting the null to the claim

`selection_divergence` = **1.0**. Picking the best null by *the null's own
outcome* selects `repeat`; picking it by *the claim arm's margin* selects
`colored` — a different process, on all three channels (`by_claim_pick_lg`,
`by_claim_pick_dw`, `by_claim_pick_cov` all `colored`). The spec's design said
this must be printed if it ever happened, and it is printed. **What it means for
the specs that used the old practice has not been said by anyone.** That is a
methodological finding about past certificates sitting in a metrics dict.

### 3.3 The constitutional picture did not move, and it is the standing drift

`coverage` EXIT 2. **0 commitments with no declared spec** — the 08-10 hole
stays shut. But:

- **4 CLAIM-DEAD** (unchanged since 09-03): smell, balance, shelter/building,
  thermal-kills. Every claim spec parked or foreclosed.
- **9 more with live claim specs and nothing passing**: touch, tool use, told
  world, proprioception, death & retry, plasticity, sleep, hunger/thirst,
  fast/slow.
- Against GOAL.md's own sentences: **one brain / unison 25 specs → 1 pass.
  Curiosity 12 → 2. Hearing 14 → 1.** These are the three the audit brief warns
  are most likely to be quietly neglected, and they are.
- **3 park→release pairs with no walkable revival path**: BA.02→LT.08,
  SH.01→SH.02, SM.02→SM.03.
- **QUEUE DEPTH: 6 dispatchable, all 6 VOID → 0 FRESH dispatches at any cost
  class.** Three classes newly empty with no path in.

### 3.3b `coverage` has printed a routing pointer at a closed desk for five days

`LC.03`'s `VOID-FORECLOSED` declaration ends:

> *"The repair is a REDESIGN of the screen or of W0, **on the owner's desk since
> 2026-08-24** (`docs/DECISIONS_NEEDED.md`, `D10`)."*

`coverage` prints that string verbatim in its foreclosure block on every run,
including this one. **`D10` fired its armed default on 2026-09-01** and is
recorded in `DECISIONS_RESOLVED.md:109`. So for five days the tool has told
every reader that a foreclosed spec's repair is waiting on the owner, when the
owner's desk closed on it and the repair is now unowned. Source is
`REVIEW_QUEUE.md:784`, not the tool.

This is the *foreclosure* analogue of the `PARK-ON-AN-UNREACHABLE-RELEASE` class
`coverage` already counts: a park whose release cannot be walked is caught, and a
foreclosure whose stated repair path points at a resolved decision is not.

### 3.4 `PROGRESS.md` states as fact something false, and coverage is red on it

The completeness audit says the four registered `GENERALITY.md` barriers
*"(`GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`) are on `coverage`'s
`GOAL_UNRUNNABLE_BASELINE`."*

They are not. `coverage.py:263` reads
`GOAL_UNRUNNABLE_BASELINE = frozenset({"DP.02", "DP.03", "LC.04"})`. Those four
GEN ids are the **live `new_unrunnable_citation` red** — GOAL.md cites them in
the present tense and all four went `welded<-LC.07` when
`lc07-checkpoint-branch` was dispositioned this morning.

The routing is right: `goal-cites-four-specs-that-resolve-to-corpses` was routed
2026-09-02, is OPEN, DUE 2026-09-10, and its own text predicted this
(*"the four citations go live the instant `lc07-checkpoint-branch` is
decided"*). The **page** is wrong, and a reader trusting it would believe an
accounted red where there is a live one. Baseline is shrink-only; the repair is
GOAL.md's text or the revival, never an addition.

### 3.5 What the builder worked on, and what it serves

Twenty iterations, and I could trace every one to a GOAL.md sentence:

| work | serves |
|---|---|
| `W1.00`, `W1.02` registered + run | *"learning its world by living in it"* — whether the world can be learned in at all |
| `EpisodicMemory` floor repair; ME.1/4/5/9/10 + XL.00 re-buys | *"he remembers the ladder, he remembers you"* |
| `me1_floor_probe` A5 bakeoff | same, by measurement instead of argument |
| `D1.0` gate execution + attempt-2 dispatch | `T2.01`, frees 34 — the unified-brain family |
| `LG.00` re-buy a7 | *"not an LLM in a costume"* |
| `T0.35`/`IMPL_DEPS` sweep, `UB.9` re-buy, ratchet arithmetic | harness integrity — no GOAL sentence, and correctly claims none |

**No drift.** The `T0.35` family is scaffolding and is registered as scaffolding
(`49e8356` deleted a `COVERS:` line that claimed a commitment which does not
exist, and took a FAIL row to do it — that is `T0.21` policing flattery and
working).

---

## 4. Is the builder alive and productive? — YES, emphatically

- **20 iterations in the last 24 h, 20 ended `rc=0`, zero aborts, zero paused
  slots, no credit exhaustion** (gate read `week:all models` 27→31% and was
  named in every slot).
- Demonstrated **105 → 106** net, and the path is the honest kind: 107 at 14:07,
  **down to 106** when `ME.3` FAILed and the builder wrote *"CHECKLIST honestly
  dropped 107→106."*
- Two failure modes handled well and worth crediting:
  - The **17:07 slot died** leaving 929 uncommitted ledger lines. The 18:07 slot
    verified they were exactly the 18 promised re-buys, committed them as found,
    then found a fault the dead slot never saw (`unreachable` 95 above floor 94),
    **bisected across worktrees rather than guessing**, and repaired it by
    unblocking `UB.9` rather than raising the floor.
  - `D25` was opened this morning, **refused by `decisions.py` as
    `MEANS-ESCALATED`**, deleted, and replaced with the bakeoff. The instrument
    beat the prose. That is the `D1` disease caught in under an hour instead of
    twenty days, and it is the single best thing in today's log.
- One flag, already self-reported: `17:17:25 MEMORY 1904980 — peak rss 2912 MB
  over the 1536 MB ceiling`. Named, not killed, per `D18`'s fired default. On a
  box with paying tenants that is the right behaviour and the ceiling question
  is the owner's.

---

## 5. Compute honesty — CLEAN

- **GPU, week 2026-W36: 5.61 h charged across 2 jobs** against Kaggle's 30 h,
  which resets today. `D1.0` phase-3 is in flight (~8.6 h est from 13:57).
- **I verified the in-flight kernel independently rather than believing the
  log** — `kaggle kernels status jack-ladder-1788703032` →
  `KernelWorkerStatus.RUNNING`. The watcher (pid 1775588, up 10 h 18 m, 7 s CPU
   — a poller, not a hang) is declared and alive. **The `T0.11` deferral behind
  an occupied venue is honest**, and the reasoning is good: 52 historical Kaggle
  jobs, zero overlapping intervals, so a concurrent submission would measure
  Kaggle's scheduler rather than the failover path.
- **No GPU hours without a ledger entry to show for them.**
- CPU today: **7,066 s**. Top items `T1.01` 2,084 s, `XL.00` 1,166 s,
  `lg00_llm_pass` 839 s, `W1.02` 636 s, `W1.00` 548 s. Note against the Review's
  09-04 finding that a full day billed *"zero seconds of new science"*:
  **today spent 1,184 s on `W1.00`+`W1.02`, the first fresh dispatches since
  `SO.08`.** The treadmill is still most of the bill, but it is no longer all of it.

---

## 6. Stuck decisions — `decisions --check` EXIT 0

7 armed, **none overdue, none `MEANS-ESCALATED`, none `UNDECLARED`**, ratchet
0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished-owner-ask. `D17` fires
tomorrow (09-07); `D22` 09-08; `D23`/`D24` 09-11; `D19` 09-14; `D20` 09-18.

Nothing on the owner's desk today has enough evidence for the system to have
settled it itself — I checked `D24` specifically, and its arithmetic (526
wall-hours against 30 h/week) is not a measurement question. **No owner decision
was quietly acted on without being recorded**; the `D25` episode went the other
way, which is the correct direction.

---

## 7. Bakeoff hygiene — CLEAN this week, one standing scar

The A5 floor bakeoff (`ab857b1`) is the week's only new bakeoff and it is
disposed correctly: six arms × 3 seeds through the real harness, a winner that
passed all four scar shapes on every seed, **and deliberately NOT adopted** —
`EpisodicMemory.py` untouched, zero certificates staled, `ME.3`'s FAIL left
standing as a true measurement, A5 routed to the Review as design input because
adopting it would restructure a call-site contract. A winner held back for the
right reason is better hygiene than a winner shipped.

No VOID was treated as a verdict this week, and no winner was chosen inside a
noise margin.

**The standing scar is unchanged and it is on the owner's desk.** The Learning
core seat is held **BY VERDICT** — the strongest marking in `CHAMPIONS.md` —
off `LC.03` = **VOID**. `champions` names it `VERDICT-IS-A-VOID` and
`TRIGGER-UNREACHABLE` (all three re-open doors closed: `LC.07` PILOT-BLOCKED,
`LC.03` VOID-FORECLOSED, `UB.10` VOID). That is `D24`.

---

## 8. The honest summary

**Yes, closer — and unusually, I can point at the thing that made it closer
rather than at the count.**

The scoreboard moved 105 → 107 → 106. The 106 is worth more than the 107,
because the subtraction was `ME.3` measuring a real cost that the project would
rather not have paid. And underneath that arithmetic, three things happened
today that are about Jack rather than about bookkeeping. His memory stopped
confabulating: it answered confidently and wrongly on **100%** of absent-target
cues at 06:51 and abstains on **100%** of them at 14:19, with recall unchanged
to four decimal places. The repair for the project's largest blocker —
`T2.01`, frees 34, FAILed since 08-12 — is computing on a P100 right now under a
gate that got strictly harder this morning. And for the first time since
`SO.08`, real compute went into asking whether the world Jack lives in can be
learned in at all, rather than into re-buying certificates.

The reason the verdict is ON TRACK and not better is §3.1. This project's
strongest habit is that it measures instead of arguing. Today it did that, and
then the measurement arrived *after* the decision it was ordered to inform and
landed on nobody's desk. `W1.00` says the honest null does not rescue seven of
eight shallowness findings — which means the case that W0 is too shallow is
stronger tonight than it was this morning, at the exact moment the desk
concluded a third of that case was really about our instruments. Every organ
here will report green on that. `fail_unowned` reads 0. `review-queue` reads 0
violations. The gap is not in the ledger; it is that a fresh FAIL can contradict
a fresh disposition and no number anywhere goes red.

The second reason is the one the Review named better than I can and I am
repeating it because repetition is the only instrument it has: **fourteen named
barriers stand between this creature and generality, ten have no spec at all,
none has a passing one.** One brain in unison — the sentence GOAL.md opens with
— has 25 specs and one PASS.

---

## FOR THE BUILDER

1. **Route `W1.00`'s FAIL, and route it as evidence against `w0-too-shallow`'s
   Pile A, not as a spec repair (§3.1).** It is attempt 1, unsettled, and
   `fail_unowned` will never see it. The row should carry the seven shift
   ratios (0.022–0.084), the `dwell` exclusion and *why* it is an exclusion
   (`f_dw` 0.0082 > its own std 0.0062 — "cannot tell", never "did not move"),
   and the plain consequence: **the Pile A / Pile B split published 09-06 is
   falsified for Pile A on its own pre-registered test.** Do not re-run `W1.00`
   to get a different branch; it fired a pre-registered branch and that is a
   result. Next free `DUE:` is **2026-09-15** (`review-queue` computes it;
   09-13 already carries 10 against a measured capacity of 1/cycle).
2. **Say what `selection_divergence` = 1.0 means for the certificates that used
   the old practice (§3.2).** Picking the null by the claim arm's margin selects
   a different process than picking it by the null's own outcome, on all three
   channels. `W1.00` was required to print it and did. Nobody has said whether
   any recorded margin was chosen that way. This is a scoping question — *which
   registered specs picked a null against a claim arm?* — answerable by reading
   the specs, and it is worth one slot.
3. **Correct `PROGRESS.md`'s baseline claim (§3.4)** — `GEN.02/03/06/09` are
   **not** on `GOAL_UNRUNNABLE_BASELINE` (`coverage.py:263` holds exactly
   `DP.02`, `DP.03`, `LC.04`); they are the live `new_unrunnable_citation` red.
   The page is the Review's to rewrite, so put the correction on the
   `goal-cites-four-specs-that-resolve-to-corpses` row (OPEN, DUE 09-10) where
   the next FULL will read it. **Never** add them to the baseline — it is
   shrink-only and this is exactly the "repair" that lowers its own number.
4. **Make `lib_seal.sh` stamp the ledger it commits from a dying run (§1.1).**
   Today it committed `experiments/ledger.json` alongside a bannered
   `PROGRESS.md` and said so only in `review.log`. A row committed by a timed-out
   organ should be as visible in the ledger as `+dirty` makes a dirty tree.
   Cheap, and it costs nothing when runs complete.
5. **Fix `LC.03`'s stale repair pointer (§3.3b).** `REVIEW_QUEUE.md:784` says the
   repair is *"on the owner's desk since 2026-08-24 (`D10`)"*; `D10` fired
   2026-09-01. `coverage` prints that sentence on every run. Re-point it at the
   real owner or say plainly that it has none — **do not delete the sentence**,
   which would convert a wrong owner into no owner silently. Worth considering
   alongside it: `coverage` counts `PARK-ON-AN-UNREACHABLE-RELEASE` but has no
   equivalent for a FORECLOSURE whose stated repair path leads to a resolved
   decision, which is why this sat visible and uncaught for five days.
6. **Give `ME.9`'s `MIN_DISTRACTOR_EVAL = 9` a reason or raise it to 30 (§2.3).**
   Its three siblings, set in the same commit, all use 30. Either is fine; the
   asymmetry undocumented is not.
7. **Standing, verified still correct as of 18:37:** `T0.11`'s re-buy stays
   deferred while `jack-ladder-1788703032` is RUNNING — I confirmed the status
   directly against Kaggle, and the reasoning (zero historical concurrent
   kernels) holds. Take it the moment the venue frees; it is 33 days old and the
   oldest live certificate in the ledger.

## FOR THE OWNER

1. **The "is W0 too shallow to learn in?" question got sharper tonight, and it
   currently has no owner (§3.1).** This morning the Review told you a third of
   the case for "yes" was really a case about weak instruments. **Four hours
   later the spec written to test that measured the opposite:** a stronger null
   does exist, and re-scoring the eight shallowness findings under it moves
   seven of them by 0.022–0.084 of their own standard deviations, and cannot
   read the eighth. The instruments were not the problem.

   I am flagging this rather than asking you to rule, because **the honest
   answer is that nobody currently holds this question.** `D10` fired its armed
   default on 2026-09-01 and is closed. `w0-too-shallow` was dispositioned this
   morning on the reading `W1.00` has now contradicted. `W1.01` — *passivity
   dies*, the spec that would actually settle whether W0 has any headroom to
   learn in — is designed, is **not registered**, and waits on
   `w1-world-edit-window` (DUE 09-13). No decision entry, no ratchet and no
   gate is currently red about any of that. If you want one thing from this
   audit, it is to know that the load-bearing question about Jack's world is
   between desks, not on one.

2. **`D24` (Learning core, `decide_by` 2026-09-11) is unchanged and I concur
   with the Review's recommendation (iii).** 526 wall-hours against 30 h/week is
   arithmetic, not a measurement, so rule 3 does not reach it. I add one fact
   from my own instruments: the seat is held **BY VERDICT off a VOID**, and all
   three of its pre-registered re-open triggers are closed doors. It is the only
   seat in `CHAMPIONS.md` carrying both `VERDICT-IS-A-VOID` and
   `TRIGGER-UNREACHABLE`. Declaring it `VENUE-UNAFFORDABLE` does not fix that;
   it makes it printable, which is the most that can honestly be bought here.

3. **The number I would want you to see, if you read one: 0 FRESH dispatches at
   any cost class.** All six specs dispatchable today are VOID rows needing a
   repair. Three cost classes are empty with no path in. Forty live
   `REVIEW_QUEUE` rows, +35 net over seven days, drain **UNBOUNDED**, ten
   promises stacked on 2026-09-13 against a measured capacity of one per cycle.
   The builder ran 20 clean iterations today and this is not its fault — it is
   `D22`, on your desk, default firing 2026-09-08.
