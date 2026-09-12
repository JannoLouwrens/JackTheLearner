# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 + Part 2.5. Part 2, the anatomy audit and the
> completeness audit are Sunday work and did not run.)

**2026-09-12 06:37–07:0x UTC — DAILY.** Window: the last 24 hours
(2026-09-11 06:37 → 2026-09-12 06:5x).

*The one sentence: **for the first morning in three, the day's headline is a
finding rather than a retraction of one — and the two findings turned out to be
the same shape as the one before them, which is a gate whose stated meaning and
whose computed quantity are different things.***

> This page is the RECEIPT for commits that already exist. Every act below was
> committed as it was made, before this page was written: `d61a11b`, `adfe6f4`,
> `bc9c5ec`, `ee122e2`, `3eb3648`, `2f8c6e7`, `e26c71e`, `f459598`, `f3dd2e7`,
> `ca09bd9`.

---

## The numbers

| | today | yesterday |
|---|---|---|
| demonstrated / registry | **108 / 245** | 108 / 245 |
| pass rate | **44.1%** | 44.1% |
| net demonstrated | **0** (4th consecutive) | 0 (3rd) |
| rework rate | 77.6% | 77.6% |
| unreachable (shrink-only) | 93 of 245 (38%) | 93 |
| ledger settlements in 24 h | **0** | 0 |
| commits in 24 h | 11 — **0 from the builder** | 19 — 0 from the builder |
| consecutive skipped builder slots | **94** | 70 |
| `week:all models` | **75%** (line 72%, elapsed 72%) | 72% (line 63%) |
| `week:Fable` | **100%** | 100% |
| GPU, live week `2026-W36` | **17.72 h of 30, 12.28 left — EXPIRES TONIGHT** | 17.72 of 30 |
| queue violations | 0 | 0 |
| live queue rows | 38 → **39** (one routed) | 38 |

Fifth day at 108/245. No builder commit since 2026-09-08T08:23.

---

## Part 1 — did the builder produce, thrash, or stall?

**Still switched off, day four, 94 slots.** Nothing new is wrong with the
outage and I have nothing to add to it that is not arithmetic. What is new is
that **this desk spent the day on the code instead of on its own retractions,
and that is where the day's two findings came from.**

### The blackout, as a bound and not a date

The 88th audit's lesson binds: quote the flat-meter bound with its assumption,
never a point forecast. Today's `06:07` reading — meter **75%**, elapsed
**72%**, line **72%**. **The gap is 3 points, down from 9 yesterday.**

- The line gains **0.375 points/hour**. This term is not a forecast — it is a
  function of elapsed time and nothing else.
- **Break-even is therefore a meter draw of exactly 9 points/day.**
- Recorded daily draws: **+21, +36, +9, +4, +3**. **Two of the five are far
  above break-even.** That is a spread, not a trend, and it is why no single
  release date is defensible.
- At yesterday's draw the gap closes ~09-12T18:00; on a flat meter ~14:00; at
  any sustained draw above ~7.5 points/day it does not close before the week
  resets.

**The only non-forecast on this page: the week resets 09-14T04:59 UTC and that
releases the builder unconditionally.** Everything above is about whether it
wakes sooner. Both the late bound and the guaranteed one land it inside W37's
fresh 30 GPU-hours — which is where the builder's own plan aimed before either
oversight organ interfered with it.

**W36 expires at the end of today.** 12.28 h left against a 17.61 h attempt. It
does not fit, and the prohibition on scraping it out still binds under every
branch.

---

## The day's science — three rows due, three RULED, none re-dated blind

### 1. `LG.03`'s liveness gate — the gate's maximum is not 1.0, and the run already knows it

`lg03-blind-twin-cannot-prove-itself-alive` offered four options. **I refused
all four**, because the defect is one line upstream of them and it is algebraic
(`d61a11b`, DISPOSITIONED, DUE 09-14). Read from source:

```
502-508   for vv, a in rec:  calib_X.append(vv); calib_Y.append(a)   # NO `if hit`
517-521   hits = [_satisfies("approach", _rollout(w, st, calib.policy(kind), ...))]
671       if m["blind_calib_rate"] < CALIB_MIN: return Status.VOID
```

The calibration tape is recorded from the privileged planner's rollout
**whether or not it succeeded** — so on a start where the servo missed, the
twin is trained to imitate a miss, and then scored on task SUCCESS.

> **Perfect reproduction of the training tape scores `planner_own`, not 1.0.**
> The gate's stated meaning is reproduction fidelity; the quantity it computes
> is `fidelity × teacher competence`. With `planner_own` = 1.00 / 0.75 / 0.75
> against `CALIB_MIN` 0.75, seeds 1 and 2 have **exactly zero margin** — the
> gate is clearable there only by flawless imitation, and on any seed where the
> servo read below 0.75 it is **un-clearable by construction.**

**The sharpest fact: `own_hit[calib_cell]` is computed on line 501 and never
read.** The run measures the number that invalidates its own gate, reports only
its mean over all cells, and throws the relevant one away.

**The repair is the only one of the five that is a TIGHTENING:** emit
`planner_calib_reach`; add `PLANNER_CALIB_MIN = 1.0` as a new VOID conjunct
checked **before** the twin's reading, indicting the teacher first; leave
`CALIB_MIN` at 0.75 absolute and `_Blind.KINDS` unchanged. Option (i) was
refused as **a loosening of a CONTROL's alive-proof** (0.5625 < 0.75), which
the one law forbids me. Option (ii) was refused as venue-selection — conjunct 3
buys (ii)'s entire benefit without the hazard, because a bad cell VOIDs the run
instead of being swapped for a better one.

**The cost, stated because it is the expensive half:** `LG.03` in W0 as built
now VOIDs on 2 of 3 seeds — **more often than today, not less** — until the
fixture admits a calibration cell the servo aces. That is option (iv)'s venue
reading, arrived at from the gate rather than asserted about the world.

### 2. `SM.03`'s arm pick — REFUSED on ordering, which is not the same as slipped

Third dated promise on this row, and I did not date it a fourth time blind
(`bc9c5ec`, DISPOSITIONED, DUE 09-15). The row has treated F1 (the saturated
split) as the decision and F2 (the dead alive-proof) as a rider since 08-30.
**That ordering is backwards:** `vis_open` reads **0.1167 against a 0.60 floor
with chance at 0.125** — *below chance*. The registered run is VOID on F2
whatever F1 does.

**All three offered arms are F1 arms, and each moves F2 the wrong way or not at
all** — shrinking `N_TRAIN_L` lowers the visual baseline's ceiling; widening
`SRC_R_RANGE` changes the visual task unmeasurably; bearing-sector holdout is
the largest generalisation demand of the three. **There is no arm among them
whose selection produces a valid run, and the only move that would "fix" F2 by
choosing is lowering `VIS_OPEN_MIN`, which I may not do.**

Ordered instead: a **scratch probe, explicitly not a pilot** (`coverage` marks
`SM.03` PILOT-BLOCKED and forbids one), in the `lg03_blind_twin_probe.py`
idiom. Its third number — `vis_open` recomputed on a split built without the
`MIN_SEP_M` exclusion — decides whether F2 is a symptom of F1 or a venue fact,
and those two answers have very different prices.

### 3. `HR.5` — the contract was closed, and it was missing the item that mattered

`hr5-fixture-refuted` is now **HELD behind `w1-world-edit-window`**
(`ee122e2`, `3eb3648`). Nine days OPEN across three promised dates, two broken
by this desk, **while its own first entry declared the dependency in prose and
never used the machine-readable field that exists for it** — its two structural
siblings were already held behind that same blocker. Relabelling a
twice-slipped row into a status that exempts it from ageing is exactly the move
this file warns can turn the bundling rule into "a place rows go to die", so it
is named rather than quietly made; the defence is that the row leaves this desk
**finished**, not quiet.

Items (1)–(4) adopted verbatim. **The fifth is the one I owed it:**

> **`four_class_audio_separability > position_only_acc` on every seed**,
> pre-registered. Items (1)–(4) all add SOUNDS; not one makes the headline
> number mean anything. The row's own metric note records that 0.583 is
> uninterpretable because the position-only control read **0.708**. **Without
> item (5) the four voices get built, the fixture reports 4/4 classes present,
> and the number everyone reads is still one its own control beats.**

### And then the three findings turned out to be one finding

`PL.02` (yesterday), `LG.03`, `HR.5` — **three specs, two days, the same
shape**: a threshold calibrated against the gate's STATED meaning while the
code computes something else. `r2_ua` was the subtrahend of its own effect
size. `blind_calib_rate` is fidelity × teacher. `four_class_audio_separability`
is the separability of position.

**`run_spec` checks the bar. The overseer audits whether a threshold MOVED.
`coverage` audits whether a commitment has a spec. `review_queue` audits
whether a promise was kept. Not one of them asks what the number IS.**

Routed as `gates-that-measure-something-other-than-what-they-say` (`ca09bd9`,
DUE 09-20, a Sunday because it is Part 2 work and `t310` already taught me this
week what filing FULL-sized work onto a DAILY costs). **All three instances
were found on VOID or FAIL specs, where somebody was already hunting for a
reason the run did not count. The dangerous case is the inverse — a PASSING
spec whose gate clears a bar for a quantity it does not name — and 108 of 245
specs are PASS with none examined for it.**

---

## Part 2.5 — steering maintenance

**1. Priorities — reconciled, in two commits, and one of them corrects me.**
Items 1–5 of the live block are unchanged and keep their order; today's three
rulings were **added** as items 6–8 (`f459598`), not substituted for anything.
The release forecast was **re-stated as a bound** (`e26c71e`) — superseded for
SHAPE, not withdrawn for error: the old range was defensible, it was just in
the point-forecast form the 88th audit's lesson deprecates, and its numbers
were a day stale.

**And a count I had wrong.** Yesterday's page told the builder there were
**three** overdue armed defaults (`D22`, `D18`, `D26`). **The instrument prints
five** — `D24` (decide_by 09-11) went overdue at midnight and `D23` is overdue
this morning. Neither appeared on my page. Corrected in the steering with the
general fix rather than the specific one: **take the count from
`python -m experiments.decisions`, never from a page.** This is the third day
running that something in this desk's prose was wrong where the instrument was
right; today it cost a count rather than a dispatch, and I found it by running
the tool instead of quoting myself.

**2. Field watch — nothing to consume.** Last sweep 2026-09-07 (wk6, `3b68b7d`),
consumed in full by the 09-07 Review. Next sweep Monday 09-14. File unchanged
since, byte for byte.

**3. Seat staleness — carried, nothing new, and for a reason.** `Learning core`
holds 3 trigger debts (`LC.07` PILOT-BLOCKED, `LC.03` VOID-FORECLOSED, `UB.10`
VOID), `World` reads no deciding run and no `TRIGGER:` declared,
`Fast/slow coupling` is welded behind `LC.03`, `D1` (Control architecture)
remains VACANT. **No seat's arena context changed in 24 h because nothing ran** —
which is the correct reading of an unchanged champions report during a
blackout, not a clean bill. The missing language-**routing** seat found on 09-10
stays routed to Sunday's anatomy audit.

**4. Organ liveness — all four alive.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 06:07 today | alive; **94 of 94 slots fired and refused** |
| overseer | 6-hourly | 06:37 today (89th, running concurrently with this page) | alive; the once-daily commit rhythm is `overseer.sh` pacing working as designed |
| field watch | Mondays | 09-07, consumed | alive; next 09-14 |
| review | daily / Sun FULL | this run | alive |

**The cross-organ write race did not recur, because I changed what I do rather
than hoping.** The 89th audit is holding `docs/DECISIONS_NEEDED.md` and
`docs/OVERSIGHT.md` dirty as I write. **Every one of my ten commits today used
`git commit --only <path>`**, which commits the named paths rather than the
whole index — the repair the 88th audit routed as its B6 after my `57f67e6`
swept its staged files. I verified before writing this page that neither of its
files is staged and that nothing of its work is in any commit of mine.

**One correction to this file's own contract, and the direction is unusual.**
`REVIEW_QUEUE.md`'s header said *"a hold whose blocker has been dispositioned is
itself a violation"* — written before `DISPOSITIONED` became a formal status OF
THAT FILE. **Read literally it forbade the `HR.5` hold I made today**, which is
legal and which `review_queue.py:487` permits (it tests `TERMINAL`, which is
`ACTED`/`DECLINED` only). Corrected in place with the provenance (`2f8c6e7`),
along with a second clarification found the same way: `HELD` buys exemption
from **STALE**, never from **OVERDUE**, so the hold needed its clock re-armed
in the open (`3eb3648`) rather than dropped. **The prose was wrong and the
instrument was right** — the opposite of this week's direction, and worth
recording for that alone.

---

## Dispositions committed this morning

**Three rows due. Three RULED — none re-dated blind, and none disposed on
capacity. Queue 0 violations before and after.** One row routed.

- **`lg03-blind-twin-cannot-prove-itself-alive` → RULED, DISPOSITIONED
  (`d61a11b`), DUE 09-14.** DISPOSITIONED and not ACTED deliberately: the
  ruling exists, the spec edit does not.
- **`sm03-heldout-split-saturated` → RULED, DISPOSITIONED (`bc9c5ec`), DUE
  09-15.** The owed unit changes hands and kind — the builder's diagnostic, not
  this desk's pick.
- **`hr5-fixture-refuted` → HELD behind `w1-world-edit-window` (`ee122e2`),
  backstop DUE 09-16 (`3eb3648`).** Contract closed, fifth item added.
- **`gates-that-measure-something-other-than-what-they-say` → ROUTED
  (`ca09bd9`), DUE 09-20.**

**Ratchets, said out loud as the tool requires:** `review_queue_net_arrivals`
**5** (MOVED −24 since 09-08) and `review_queue_piled_on` **8** (MOVED +2).
The net-arrivals figure went **4 → 5 by my own hand** when I routed the sweep
row — said plainly rather than left for the tool, because a desk that routes
work to itself and reports only the ratchet's fall is flattering itself. The
`+2` on `piled_on` predates this run; today's dates (09-14, 09-15, 09-16,
09-20) all sit under the measured capacity of 6.

---

## The frontier

**`D1.0`'s twin-spread probe** is still the most important unblocked unit
(forward passes only, both branches pre-registered, DUE 09-14), standing in
front of `T2.01` — frees 35 / blocks 38, the largest single unblock in the
project. Untouched; nothing ran.

**What changed at the frontier today is nothing, and what changed behind it is
three specs' worth of gate.** The constraint is the same single gate it was
yesterday, now with a 3-point gap instead of 9 and a guaranteed release 47
hours out. The board the builder wakes to is three units longer than it was,
and every one of the three is CPU work that does not compete for the GPU
window its first item is waiting on.

---

## The honest paragraph

No numbers. Jack is exactly what he was on Monday, for the fourth day — and
this is the first of those four days that did not end with this desk correcting
something it had itself published. That is worth naming precisely, because the
difference was not care or luck: it was where I pointed. The last two mornings
were spent auditing our own prose, and both found errors in it, and both of
those errors were mine. This morning was spent reading the code that three
queue rows were arguing about, and it found that all three arguments were being
had one level too high — the options on offer were about which repair to buy
while the defect sat underneath all of them, in what the gate actually
computes. That is the week's single most important step toward Jack, and it is
not a capability: it is that the creature's tests can now be wrong in a way we
know how to look for, three times over, with a name and a date and a row. The
most concerning drift is the shadow the same finding casts. Every instance was
caught on a spec that had already failed, where somebody was hunting anyway.
The gates nobody has read are the ones that passed, and a passing gate that
measures the wrong quantity is not a red mark waiting to be found — it is a
certificate this project believes, that the builder builds on, that the
overseer will check for a threshold that never moved, and that every organ here
will go on calling green. We have spent four months learning to be honest about
whether a number clears its bar. We have never once asked what the number is.
And the encouraging thing, stated last because it is small and real: the
correction I made to this file's own contract today went the other way for the
first time this week. The instrument was right and the prose was wrong. That is
the direction I would like the drift to run.

---

## FOR THE BUILDER

**Your first act on waking is still a measurement, not a spec** — append one
line to `docs/LOOP_JOURNAL.md` recording how many consecutive slots you skipped
and the `week:all models` reading that released you. It is **94** and counting.
Nothing in this system counts a dark slot. Then, in order:

1. **YOUR W37 PLAN IS RIGHT AND IT IS UNCHANGED.** W37 opens Sunday 09-13, as
   you wrote. **`W36` expires at the end of TODAY** with 12.28 h left against
   attempt 2's measured 17.61 h. If you wake this afternoon you will be looking
   at a pot that does not fit and a deadline that expires tonight: **do not
   scrape the attempt out of it.** The precondition binds under every branch —
   twin-spread result on the row, successor gate committed in a non-dispatch
   commit, and only then a dispatch. An unchanged re-dispatch stays forbidden.

2. **FIVE overdue armed defaults are yours to fire, not three — and take that
   count from the tool, not from this page.** `D22` (09-09), `D18` (09-10),
   `D26` (09-11), plus **`D24` and `D23`, which I failed to print yesterday**.
   Run `python -m experiments.decisions`; it prints `OVERDUE — DEFAULT IS DUE
   TO FIRE` beside each. Use the required journal wording: *"the owner did not
   rule by <date>, so the pre-registered default fired."* Note `D24`'s option
   (ii) is a THRESHOLD MOVE and may not fire by silence — its default is not
   (ii).

3. **`PL.02`'s eye gate is RULED — implement it** (`5e39771`, DUE 09-14).
   Carried unchanged from yesterday. Raw-pixel radius ridge R² ≥ 0.80,
   `EYE_RADIUS_R2_MIN` unmoved, measured on the run's own probe episodes;
   `r2_ua` stays a first-class recorded metric. Then a smoke.

4. **`LG.03`'s liveness gate is RULED — implement it** (`d61a11b`, DUE 09-14).
   Emit `planner_calib_reach`; add `PLANNER_CALIB_MIN = 1.0` as a VOID conjunct
   **checked before `blind_calib_rate`**; re-run (CPU, ~725 s, 3 seeds).
   `CALIB_MIN` does not move and `_Blind.KINDS` does not change. **Expect more
   VOIDs, not fewer** — it is a tightening and it is supposed to cost.

5. **`SM.03`'s F2 probe** (`bc9c5ec`, DUE 09-15). Not a pilot; `coverage`
   forbids one. No seeds, no ledger row, no gate frozen, no constant moved.

6. **`HR.5` — nothing for you yet**, and item 8 of the steering says why. Do
   not start the world edit outside `w1-world-edit-window`.

7. **`W1.04` still gains conjunct (c) before you register it** — carried
   unchanged from 09-10. Register from the amended design, not the 09-06 text.

---

## FOR THE OWNER

**1. `D26` — unchanged, already routed, and the calm case for it got calmer.**
See `D26` and its 09-11 premise-correction addendum (`04d8b69`). My
recommendation stays **(i) ATTRIBUTE THE LINE**, on the structural argument:
`pace_gate` rations a shared meter with no attribution and no ordering, and
that recurs every time an external consumer draws, regardless of how long *this*
blackout lasts. The urgency is still withdrawn — and today's meter reduces it
further, not more: the gap is 3 points rather than 9, and there is an
unconditional release at the 09-14 week reset. **The overdue default
`(iv) MEASURE ONLY` is unaffected and should still fire.**

**2. `D22` HAS JUST BEEN FIRED. `D18`, `D23`, `D24` and `D26` remain overdue
and unfired — and yesterday I told you there were three.** Repeated verbatim
rather than dropped, because a `VANISHED-OWNER-ASK` is the scar this section
exists to prevent, and because the miscount is mine. `D24` went overdue at
midnight on 09-11 and `D23` this morning; neither was on yesterday's page.

> **CORRECTED IN PLACE, minutes after this page was committed.** I wrote that
> all five were unfired. **The 89th audit fired `D22`'s armed default at
> 06:5x this morning (`7d0b49c`), concurrently with this run** — default
> **(i) THE RULE STANDS**, which denies my own ask and writes nothing, exactly
> as I said it should be fired if nobody fired it. It is off your desk. I am
> correcting rather than rewriting because the claim was false for about ten
> minutes and a page that quietly repairs itself teaches nobody anything. Note
> what this is: **the first armed default this project has fired from a
> non-builder organ**, and the overseer's reason for reaching for it is the
> same fact that dominates this page — the one organ that normally fires them
> has refused 94 consecutive slots.

**`D24` carries the one warning worth repeating here: its option
(ii) SHRINK THE CLAIM is a threshold move and may not fire by silence.** Its
default is not (ii) and nothing in the silence should be read as choosing it.

**3. NO-DECISION: the day's science, declared because it is mine to do and not
yours to rule on.** Three rows due, three ruled, zero re-dated on capacity. The
sweep they generated —
`gates-that-measure-something-other-than-what-they-say` — is routed to my own
Sunday docket (`ca09bd9`, DUE 09-20), **not to your desk, because Part 2 test
re-examination is already this desk's standing jurisdiction and manufacturing a
fork out of my own homework would spend your attention on a decision that is
not yours.** You should know it exists because of what it may return: if the
sweep finds a **PASSING** spec whose gate clears a bar for a quantity it does
not name, the repair is a strengthening under the standing law and **that
certificate re-buys**. A PASS that has to be re-bought is the outcome the row
exists to find, not a reason to avoid looking — and it is the only foreseeable
way `demonstrated` goes *down* by an act of this desk. I would rather you heard
that from me now than from a number later.

**4. NO-DECISION: the docket, announced rather than asked about.** Queue 0
violations before and after; 39 live rows. `review_queue_net_arrivals` **5**
(MOVED −24 since 09-08) and `review_queue_piled_on` **8** (MOVED +2). The
net-arrivals figure rose by one **by my own hand** this morning. Sunday 09-13
still carries **14 rows against a measured capacity of 6**; I have said for a
week it will not clear, and I have dated nothing new onto it.

**5. NO-DECISION: liveness report, nothing here to rule on.** All four organs
alive; the builder has fired and refused 94 consecutive slots. The overseer's
once-daily commit rhythm is its own pacing working as designed. The cross-organ
write race I caused on 09-11 did not recur: all ten of today's commits used
`git commit --only`, and I verified the 89th audit's two dirty files are in
none of them.
