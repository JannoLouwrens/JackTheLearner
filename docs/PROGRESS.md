# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 + Part 2.5. Part 2, the anatomy audit and the
> completeness audit are Sunday work and did not run.)

**2026-09-07 06:37–07:1x UTC — DAILY.** Window: the last 24 hours
(2026-09-06 06:40 → 2026-09-07 06:40).

*The one sentence: **the builder discharged seven of yesterday's nine orders
inside a day and repaired the confabulation this desk found — abstention
0.0000 → 1.0000 with recall not moving a digit — and in doing so it broke a
different spec, proved by measurement that the break cannot be fixed inside the
module, and handed me back a decision I took six days early because the damage
was mine.***

---

## The numbers

| | now | 09-06 (FULL) | Δ |
|---|---|---|---|
| demonstrated / registered | **106 / 245** | 104 / 242 | **+2 / +3** |
| pass rate | **43.3%** | 43.0% | **+0.3 pts** |
| rework (attempt > 1) | 76.9% | 75.0% | +1.9 |
| settled FAIL / VOID | 24 / 13 | 23 / 13 | +1 / 0 |
| unreachable | 94 of 245 (38%) | 95 | **−1, baseline shrunk** |

**Head settlements in 24 h: 47** — 43 PASS, 3 FAIL, 1 VOID; 80 ledger events
including history. **First-ever PASSes: 16 this week against 11 the week
before.** The demonstrated set moved by exactly four specs: **gained `ME.1`,
`T0.35`, `W1.02`; lost `ME.3`.** Everything else in those 47 rows is the
certificate re-buy treadmill, and I would not want the 47 quoted as output.

**Goodhart: the rate rose and I am not going to call that a good sign on its
own.** +3 registered against +2 demonstrated is close to neutral, and the
composition is what matters: one PASS is a genuine repair (`ME.1`), one is a
new world spec (`W1.02`), one is a re-buy (`T0.35`), and the loss (`ME.3`) is
this desk's own conjunct doing its job. **Unreachable came DOWN 95 → 94** —
yesterday I raised that baseline to fence the `ME` family off behind `ME.1`,
and the repair landed inside a day, so the fence came down. A shrink-only
ratchet shrinking is the cleanest number on this page.

---

## Part 1 — did the builder produce, thrash, or stall?

**It produced, and by an unusually large margin.** Of the nine `FOR THE
BUILDER` items I published yesterday, **seven are discharged**: the adopted
`D1.0` gate executed (item 1), the similarity floor repaired (2), the
distractor conjunct landed on all four sibling specs (3), `W1.00` and `W1.02`
registered and run (4), `LC.07`'s CPU venue priced at 617.8 core-h / 38.6 days
(6), `T1.01` re-run and re-bought (7), and `T0.11` — the 33-day oldest live
certificate — re-bought at attempt 2 (8). Item 5 is dated 09-13. Item 9 was a
list of prohibitions and none was breached.

**And it did that off a page formally marked UNVERIFIED, which is the day's
quiet finding.** Yesterday's FULL was killed by `timeout(1)` at 07:17:11,
`rc=124`, and `lib_seal.sh` banners its page *"THIS IS A DRAFT, NOT A
FINDING."* The builder executed it anyway and was right to — the page was
complete and every disposition in it was already committed. Routed as **`D25`**
below. I am also correcting my own desk: yesterday's page said *"four of the
five Sunday FULL runs ever scheduled died at max turns. This is the fifth,"*
which reads as though the fifth survived. **All five died. The fifth died
against a different wall, having already banked its work.**

### The science of the day, and it cost a certificate again

`ME.1`'s repair is real and I verified it rather than reading the report:
`distractor_abstention` **0.0000 → 1.0000**, and `cued_recall` **0.85 ± 0.0136
— byte-identical to the FAILing attempt and to every attempt before it.** The
row asked for abstention *without* costing recall and the recall number did not
move at all. The trade this project was afraid of did not happen.

Then it broke `ME.3`, and the way it broke is worth more than the fix. `ME.3`'s
raw arm cues the store with `" ".join([speaker] + candidates)` — five words,
every one known to the store, mutually exclusive by construction — so the new
0.95 coverage floor abstained on **every** question: `raw_tokens_mean`
**40.0 → 0.0**, `raw_acc` 0.625 → 0.2917 against a base rate of 0.25, the
reflect arm untouched at 1.0, and the "gain" inflating to 0.708 while the
equal-tokens honesty gate refused to certify a starved null. **The harness
caught it. Nothing was hidden.**

The builder then did the thing this system is for: instead of arguing for a
floor value, it measured the statistic every floor in the family thresholds on.
**Cues that MUST abstain read `bestcov` 0.667 exactly; cues that MUST answer
read 0.400 exactly. Gap −0.267, overlap 1.000, all three seeds.** The two
populations separate in the wrong order, so **no monotone single-cue floor can
serve both** — an impossibility, not a preference. It also tried to escalate
that to the owner as `D25`, and `decisions.py` refused it (`MEANS-ESCALATED`: a
means fork is settled by bakeoff, not by authority). The checker was right, the
bakeoff ran, and arm A5 — the contract split — is the sole survivor.

### The queue, and it is mine

`review-queue` reads **40 live rows, drain UNBOUNDED**, 34 arrivals against 3
disposals over the trailing week. **Four rows came due today.** I disposed
three and re-dated two in the open (one of the three was dated 09-13 and pulled
forward). Of the nine specs whose dependencies all PASS, **all nine are held**
— four PARKED, four PILOT-BLOCKED, one on `D19` — and **eight of the nine wait
on a redesign owed by this desk.** The builder is not the constraint and I have
told it so in its own file.

---

## Part 2.5 — steering maintenance

**1. Priority reconciled.** `1'''`/`2'''` were spent — `SO.08` PASSed 09-06
12:16 and `D21`/`D16`/`D15` all fired on schedule. **Sixth consecutive day a
priority block was fully executed inside its own day.** Replaced with
`1''''`/`2''''`, which for the first time in a week hands the builder *runnable
units* rather than an explanation of why the board is empty: `ME.3`'s redesign,
the `EpisodicMemory` docstring, the `audit_supersedes_fail` truthfulness fix,
`PL.00`'s renderer bakeoff, and one free standing rule. `3''` unchanged.

**2. Field watch wk6 consumed — 42 minutes after it landed**, the shortest
sweep-to-disposition gap this project has recorded. All three nominations
accepted, **none as an arm**. `N1` (Context Collapse on `A4`) accepted as a
mandatory diagnostic — **on our own reason, not the papers'**: `A4` was seated
on `life_gain`, a number an actor–critic actor produces from the model state,
which requires nothing whatsoever of the latent predictor's action-conditioning.
The seat is named for a world model and the measurement certifies a
representation. The papers assert the failure and **none of the three measures
it in public**, so they are the reason to look and never the finding. `N2`
accepted as a dated pre-registration on tomorrow's `UB.10` pick and **barred
from becoming a gate** — its decisive quantity is named but never defined, and
inventing the metric to satisfy the paper that named it is the purest form of
the Goodhart that front was refused over twice. `N3` accepted as a required
control on the `t402` bakeoff: **measurement imported, recommendation refused**
(it argues against `GOAL.md` stage 4).

**The precondition the sweep found without naming it, and it is free:** wk6
measured that **no trained `A4` weights exist on disk**. `A4` holds the
Learning-core seat. So every future question about the one arm this project
rests on costs a full retrain — 4.8 core-h per seed — before it can be asked.
Ordered: any run that seats or challenges a champion must persist its weights.
Its whole cost is disk.

**3. Seat staleness — one finding, and my own morning ruling is entangled with
it.** The **Vision encoder** seat is held **BY DEFAULT, UNCONTESTED**, and its
arena is `T2.03` + `PL.02` — and `PL.02` is precisely the spec I ruled this
morning must stay blocked behind `PL.00`'s renderer. So a default champion's
only live challenger is unreachable for reasons that have nothing to do with
vision. The renderer bakeoff I ordered is the cheapest thing that unsticks it,
and I did not notice that when I ordered it. Second: **Control architecture
(D1) is VACANT and its entire arena is `D1.0`, which has now VOIDed twice**,
burning ~16 GPU-hours each time. Learning core keeps its `TRIGGER-UNREACHABLE`
debt pending `D24`.

**4. Organ liveness — all four live, verified against `/data/jack-logs` mtimes
rather than anyone's report.** Builder 06:15 (hourly), overseer 06:37 (6 h),
**field watch 05:56 today — Monday, on cadence, second consecutive sweep on
schedule**, review 06:37 (this run). `lost_iterations.log` still 0 bytes.

---

## Dispositions committed this morning (each in its own commit, as it was made)

1. **`t027-preserved-failimpl-as-artifact` — ACTED.** `D16` fired 09-06 by
   armed default, option (b) alone: `T0.27` stays RED, untouched. The owner's
   silence chose the option that costs a visible failure over a manufactured
   green and there is nothing for me to add. Two facts survive the close rather
   than being filed with it: the violation arrival rate that argued against
   (a)'s premise has been **flat at 3 for three days**, so (b) is holding and
   `D16` is what re-opens if it climbs; and `audit_supersedes_fail` still
   prints *"that implementation was never committed"* for all three violations
   when two of them have hash-verified bytes under `refs/jack/failimpl/`.
2. **`pl02-dependency-on-pl00-verdict-vs-table` — DISPOSITIONED, (i)+(iii).**
   The edge STANDS. Option (ii) refused on the router's own reason: re-pointing
   `PL.02 → PL.00` would loosen the sole registered falsifier of the
   PLASTIC-ONLY decree, on a sentence that reads both ways, in the week that
   edge produced an inconvenient FAIL. Arm (iii) ordered instead — the eye's
   price is **fixed per-call overhead** (`render_ms_224` 39.17 vs
   `render_ms_64` 40.04, 12.25× the pixels for the same money; render-only
   4.231 below a 5.0 floor with no encoder in the loop). **A clearing arm
   dissolves the edge by satisfying it, never by editing it.**
3. **`me1-similarity-floor-never-abstains` — DISPOSITIONED six days early, and
   re-dated EARLIER (09-13 → 09-11).** A5 adopted on the −0.267 separability
   gap. `ME.3`'s harness declares its alternatives; **matched arms binding**;
   `EpisodicMemory.py` untouched; no threshold moves; A5's 0.552–0.688 against
   A0's 0.625 is a **restoration** of the raw null and I will not write it up as
   a strengthening. **One conjunct added and it is strictly harder:
   `raw_answer_rate >= 0.95`** — the equal-tokens gate caught this only because
   starvation was total; half-starvation would have passed it *and* inflated
   the gain, arriving as a better-looking result for the claim.
4. **`t310-anticorrelated-gates` → 09-11 and `sm03-heldout-split-saturated` →
   09-12, re-dated in the open, second slip on both, named as capacity.** Both
   onto one-row days rather than 09-13's pile of ten. Nothing moves meanwhile
   on either spec, in either direction. Cost stated: `SM.02` is parked on
   `SM.03` as its revival path, so smell stays a commitment with a spec and no
   measurement for five more days.
5. **Field watch wk6 consumed** into `INTEGRATION_QUEUE`.
6. **`ladder_prompt.md` priority block replaced.**
7. **`D25` routed** to `DECISIONS_NEEDED`.

---

## The frontier

`T2.01` still tops it and has for weeks: **frees 35 / blocks 38**, settled FAIL,
implementation unchanged **28 days**, repair through `D1.0` — whose attempt 2
**VOIDed at 01:57 this morning because the gate I adopted on Sunday fired on the
untrained twins.** That is the gate working: it refused to record a learning
verdict on anyone from a run whose own reference arm did not clear. **A VOID
from a gate that fired is worth more than a PASS from a gate that could not**,
and it is still the second time ~16 GPU-hours have bought no verdict. Attempt 3
does not exist until `d10-successor-rerun-under-adopted-gate` (**DUE tomorrow**)
answers why the twins failed. Behind it: `LT.01` frees 7, `NE.01` frees 5,
`UB.10` frees 4 (its pick is tomorrow), `T2.02` frees 3, `LG.03` frees 3,
`HR.1` frees 3 (D19-held).

---

## The honest paragraph

We are closer, and today the evidence is unusually clean, because for once the
thing that improved was the creature and not the instrument that watches him.
Yesterday this desk found that his memory answered confidently and wrongly every
single time it was asked about something that never happened; within eight hours
it no longer did, and — this is the part that matters — it did not lose a single
point of what it could already remember to get there. That is not a scoreboard
move, it is the difference between a companion who might invent your preferences
and one who says he does not know. But the same repair silently starved a
different question, and instead of tuning a number until both looked fine,
somebody measured the one quantity that both questions have to share and found
they demand it in opposite directions by an amount that no setting can bridge.
That is the best hour of method in the project this week and it came from the
builder, not from here. The most important step toward Jack was that the memory
learned to refuse. The most concerning drift is entirely mine and it is
structural: forty live rows, an unbounded drain, every one of the nine specs
that could run today held behind a redesign this desk owes, and four dated
promises landing on a single morning of which I could honestly keep three. The
builder has now spent six consecutive days executing everything I give it inside
the day I give it, and the reason the board keeps emptying is not that it is
slow — it is that I am the part of this system that does not scale, and no
instrument here will report that as a failure, because I am the organ that
writes the reports.

---

## FOR THE BUILDER

1. **`ME.3`'s harness redesign — the contract split, first, because it is a
   FAIL this system caused itself.** Full order on the queue row and in
   `1''''`. Two conditions bind: the reflect arm gets the **identical** declared
   shape (matched arms or no redesign), and **`raw_answer_rate >= 0.95` is
   added** as a conjunct. `EpisodicMemory.py` is not touched, the 0.95 coverage
   floor does not move, `ME.1`'s bar does not move, no `ME.3` threshold moves,
   and zero certificates stale. Do not write A5's 0.552–0.688 up as an
   improvement on A0's 0.625 — it is a restoration.
2. **`EpisodicMemory.recall` owes a docstring naming what it cannot do.** No
   certificate records a limitation, so if the −0.267 gap is not in the
   module's own words it is nowhere. The scorer cannot recover AND-intent from
   OR-intent in a token bag; callers with alternatives must declare them.
3. **`audit_supersedes_fail` prints a sentence that is false for most of its
   own rows.** Two of the three live `T0.27` violations (`LG.00`, `T0.29`) have
   hash-verified preserved bytes. `D16` ruled on the **gate**; it said nothing
   about the instrument telling the truth while red. `T0.27` stays FAIL, the
   count does not move, nothing is re-run to make it green.
4. **`PL.00`'s renderer bakeoff, arm (iii).** Frame-skip, context reuse,
   batched `update_scene`, coarser scene — CPU, against `PL.00`'s existing rig
   at its unmoved 5.0 floor. This is also what unsticks the Vision encoder
   seat's only challenger.
5. **Free, and it is a standing rule: any run that seats or challenges a
   champion must persist its trained weights as a run artifact.** Not a spec,
   not an instrument, not a threshold. Until it holds, *"we could measure that
   on the seated arm"* is false about every seat in `CHAMPIONS.md`.
6. **Standing prohibitions, unchanged and restated in `3''`:** no third `D1.0`
   dispatch before tomorrow's row answers; `HR.1`–`HR.4` stay D19-held to
   09-14; `HR.6` stays behind `HR.5`; `LF.01` attempt 2 waits for the 09-09
   design; the CPU-accountant rule stays as narrowed on 09-05.

---

## FOR THE OWNER

1. **All five Sunday FULLs have now died mid-run — but the fifth died having
   already committed everything, and the seal cannot tell those two deaths
   apart. Routed as `D25`** (`class: process`, `decide_by` 2026-09-13). The
   08-31 turn fix worked and moved the binding constraint to the wall clock:
   `review.sh` gives FULL `40m / 240 turns` and 09-06 was killed by `timeout`
   at exactly 40 minutes, `rc=124`. Before that kill it had committed six
   dispositions, two seats, the whole page and its log row. `lib_seal.sh` sees
   only `rc != 0` and banners it *"THIS IS A DRAFT... UNVERIFIED"* — correct for
   09-05, which died having appended nothing, and false for 09-06. **The
   builder then spent 24 hours executing seven of nine items off a document
   formally marked unverified, and was right to.**

   > **My recommendation: (iii) — fix the seal, and do NOT raise the wall
   > clock.** The 2026-09-06 run is the evidence: forty minutes was enough to
   > do Part 2, both completeness audits, six dispositions, two seats and the
   > entire page, and the only thing it was not enough for was saying so.
   > Buying minutes to improve an exit is spending the one resource that has
   > historically silenced this whole system, in order to fix the cheapest part
   > of the problem. The expensive part is an instrument that tells a true
   > thing about one Sunday and a false thing about the next, in the same words
   > — and that costs nothing to repair. The price of (iii), stated: Sunday
   > FULLs will keep exiting non-zero, `review.log` will keep recording
   > `rc=124`, and the organ will keep looking unhealthy to anything that reads
   > exit codes alone. I would rather have a truthful banner over a complete
   > page than a green exit code, and if the choice is ever between the two,
   > this desk should take the banner.

2. **`D24` (decide_by 2026-09-11) now gates more than it did when you got it,
   and I am telling you rather than re-asking.** Cited, not re-routed. Field
   watch wk6 nominated a Context Collapse diagnostic on `A4` — the arm that
   holds the Learning-core seat — and its cheap route bolts it to `LC.07` at
   zero marginal compute. **If `D24`'s default fires and declares the venue
   unaffordable, that route does not exist and the diagnostic costs 14.4
   core-h**, roughly 90% of a whole CPU day. My recommendation on `D24` is
   unchanged and is (iii); this note only records that its blast radius grew,
   and that I am still recommending the option that leaves the seat
   uncontestable rather than shrinking a 10× claim to fit a budget.

3. NO-DECISION: a report on this desk's own capacity, with nothing for you to
   rule on today, and I would rather you saw it before it becomes an ask.
   The review queue holds **40 live rows with an UNBOUNDED drain** — 34
   arrivals against 3 disposals over the trailing week — and **eight of the
   nine specs whose dependencies all PASS are waiting on a redesign this desk
   owes**. Four dated promises came due this morning and I could honestly keep
   three. The builder has executed every order I have given it inside the day
   for six consecutive days. **The bottleneck in this system is not compute,
   not credits, and not the builder; it is the organ writing this sentence**,
   and the two instruments that would normally catch such a thing cannot,
   because `review-queue` correctly declines to call a slow week a violation
   and the liveness check reads my log's mtime, which is fresh. If the drain is
   still unbounded a week from now it becomes an ask about cadence or scope,
   and it will arrive as one.

4. NO-DECISION: liveness report, nothing here to rule on.
   All four organs live, verified against `/data/jack-logs` mtimes rather than
   anyone's report: builder 06:15 (hourly), overseer 06:37 (6 h), field watch
   05:56 today (Mondays — **second consecutive sweep on its intended cadence**,
   and `FIELD_WATCH.md` wk6 was consumed by this run 42 minutes after it
   landed), review 06:37 (this run). `lost_iterations.log` still 0 bytes and
   still never exercised.
