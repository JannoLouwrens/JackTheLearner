# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-22 06:37–07:0x UTC — 159th audit.** VERDICT: **DRIFTING.**

Run against a builder that woke at `06:07` after a **23-slot blackout**, and
against a Review that was **live in the same file for the whole sitting** — it
committed acts 1–5 (`80fb8ee` → `3ed8d0d`) between 06:38 and 06:52 while this
page was being written. Every instrument was re-run after my own edits. **Two
of my three routed rows were swept into the Review's commits** (`5404cc3`,
`3ed8d0d`) because it staged the file by name while my appends were
uncommitted — disclosed in §9 rather than left for someone to find.

True exit codes, taken from the processes and not from a pipeline, **after** my
edits: `coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **0** (it exited **2** at my open, with 11 OVERDUE; the
Review drove it to 0 during the sitting) · `run status` **2**.

Ratchet line **at my open**, before I touched anything:

> `5 MOVED (fail_unowned_owned_forms {'disposed': 1, 'queue-row': 27,
> 'repaired_by': 1} -> {'disposed': 2, 'queue-row': 26, 'repaired_by': 1},
> review_queue_net_arrivals -5 -> -3, review_queue_piled_on 15 -> 6,
> review_queue_violation_forms {'OVERDUE': 16} -> {'MALFORMED': 1, 'OVERDUE': 7},
> review_queue_violations 16 -> 8); floors: 2 ABOVE (pass_on_dead_dependency,
> unreachable), 0 BELOW, 0 UNVERIFIED.`

And **after** my three routed rows and the Review's five acts, quoted because
part of the difference is my own:

> `5 MOVED (… review_queue_piled_on 15 -> 5, review_queue_violation_forms
> {'OVERDUE': 16} -> {}, review_queue_violations 16 -> 0); floors: 2 ABOVE
> (pass_on_dead_dependency, unreachable), 0 BELOW, 0 UNVERIFIED.`

`claim_dead = 3` · `commitments_uncovered = 0` · `dark_slots = 0` ·
`fail_unowned = 6` — all unmoved by me.

---

## RANK 1 — `D47`'s headline number was never a live reading, and its armed default fires in two days quoting it

**This is the most damaging finding available this morning, because it is about
how a clock gets its facts.** `D47` (`class conduct`, `decide_by 2026-10-24`)
asks whether the Review's queue is drainable. Its measurement section is headed
**"The measurement, read this sitting"** (2026-10-17). That is a freshness
claim, and **nothing in this repo checks one.**

I recomputed every figure with the instrument's own
`experiments/review_queue.py:throughput()` against **every October revision of
`docs/REVIEW_QUEUE.md` in git** — the same git-history basis the function
itself uses. Method validated before being trusted, against the reading the
tool printed at my open:

```
reconstruction  arrived 16 (2.29/c)  disposed 21 (3.00/c)  live 72  drain 100.8
tool printed    arrived 16 (2.29/c)  disposed 21 (3.00/c)  live 72  drain 101
```

| quoted in `D47` | status | the revision that actually printed it |
|---|---|---|
| `77 live rows` | **2026-10-13**, not 10-17 | `8aea5db`..`2adfc24` 12:26–12:37 — live 77, drain **49.0** |
| `drain 54 cycles` | **2026-10-13**, not 10-17 | `2839b0b` 12:26 — live 78, drain **54.6**. The ONLY October revision printing 50–58 |
| `39 OPEN, 35 DISPOSITIONED, 0 HELD` | **correct and fresh for 10-17** | all eight 10-17 revisions — where `live_now` = **74** and `drain` = **103.6** (172.7 earlier) |
| `arrivals 2/cycle vs disposals 1.71/cycle` | printed at **no** revision, 09-20 → today | — |
| `DECIDE:` block's `74 live rows` | **correct for 10-17** | contradicts the body's own `77` |
| `DECIDE:` block's `drain 47 cycles` | printed at **no** revision | — |

Two consequences stand on their own:

1. **The entry's own breakdown sums to 74** (39+35+0) **while its headline says
   77**, in the same section.
2. **`arrivals 2/c vs disposals 1.71/c` is not merely stale, it is
   self-inconsistent.** Arrivals above disposals makes `net >= 0`, which
   `review_queue.py:510` renders as `unbounded` — never as a finite cycle count.
   The quoted pair and the quoted `drain` cannot have come from one reading.

**Why it matters more than an arithmetic slip.** The armed default's stated
price is *"the drain reading goes on printing 54 cycles."* The live reading at
my open was **101**, and after today's routing **173**. **The error direction is
always toward health:** the stale figure makes the status quo look roughly twice
as survivable as measured. The overseer fires armed defaults on the first legal
day — so without this correction I would have fired a default on 10-25 blessing
a hold at half its measured cost.

**What does NOT change, said plainly because fairness is part of the job.**
Every *qualitative* claim in `D47` survives and most get stronger: the queue is
not drainable at the desk's measured rate, re-dating is the only repair left in
volume, and recommendation **(ii)** — split the desk's docket from the
builder's — is independently confirmed below. Nothing here argues against
`D47`. It is `D47`'s own case, priced correctly.

**ACTED, not merely reported:** an `EVIDENCE ADDENDUM` is appended to
`docs/DECISIONS_NEEDED.md` in the established `D40`/`D46` idiom — no option
picked, no `decide_by` moved, nothing fired, `class`/`default` byte-untouched.
The **class** is routed (§9, row 1).

## RANK 2 — `OVERDUE 27 → 0` did not survive the night, from rows the desk could see

`d5e4fd5` (2026-10-21 06:51) published the class at **0**. At my open it read
**11**. Nothing was smuggled and no date was moved: all 11 already carried
`DUE: 2026-10-21`, written on 10-08, 10-10 and 10-17, and the file's own date
table printed **"2026-10-21  11 rows"** that morning. A row due *today* is not
overdue — so the 0 was true when measured and arithmetically certain to become
11 within hours.

**The desk is not accused of hiding it.** `d5e4fd5` assigned 18 new dates and
**not one was 2026-10-21**; it says so explicitly (*"dates were assigned
outside the band this sitting is itself filling"*). It knew the day was full.
What it did not do is dispose of, or name, the 11 already standing in it. **The
defect is that no instrument and no page says "N live rows are due TODAY and
untouched"**, so a desk can truthfully report 0 with tomorrow's 11 in front of
it.

**3 of the 11 were finished work awaiting only a signature** — the file's own
`DELIVERED — AWAITING STAMP` block named them with commits
(`oversight-for-the-builder-has-no-reader` `c8ee77b`, `lc03-five-controls…`
`da9ece0`, `t306-random-arm…` `26d0541`). That is `D47`'s recommendation (ii)
measured in the live file.

**CREDIT WHERE IT IS DUE, measured not assumed.** Today's sitting did better on
exactly this axis: acts 1–3 stamped those three ACTED, and act 5 re-dated the
seven remainder onto **10-28/29/30/31** — nothing onto 10-22 or 10-23. Today's
`0` is not set up to reconstitute tomorrow. The reading is still missing, which
is why the row is routed (§9, row 2) rather than closed.

## RANK 3 — the dark-slot ratchet is a STREAK, so an ENDED blackout reads zero

`dark_slots` reads **0, "unchanged since 2026-09-30"**, at its floor, with
`DARK_SLOT_GRACE = 2`. The builder lost **23 consecutive slots** between
`2026-10-21T05:27:02` and `2026-10-22T06:07:09` — **24 h 40 m**, peak streak
recorded in `ladder.log`'s own PACING line at 05:07. The 06:07 iteration ran
`rc=0` and the counter reset. **Nothing in `experiments/` now holds any trace of
those 23 slots.**

`_dark_slots` (`run.py:1972-1998`) returns `usage_attribution.dark_slot_streak`
— the trailing streak. Its own comment records the 131st audit FTB 1: the streak
*"appeared NOWHERE in `experiments/` … so a 28-slot blackout read 'healthy' on
every current-state page."* **The counter built to close that hole has the
identical blindness in the time dimension.** And every organ here starts on a
cron slot the pacer has just released — so the audit most likely to read the
counter is the one reading it right after the gate let go. The 158th audit saw
`0 → 7` only because it ran at 12:55 mid-blackout; I saw 23 only by reading
PACING lines by hand.

**The bias is not symmetric.** A streak counter can over-report nothing; its
only failure mode is reading 0 for a week in which hundreds of slots were lost
— exactly the quantity `D30`, `D40` and `D46` have each been unable to settle,
all three arguing attribution from hand-read logs because no counter
accumulates. Routed (§9, row 3).

**The pacing itself is acting correctly on its inputs and I am not indicting
it.** `ladder.log` 05:07: `week:all models` **53%** at **43%** of the week
against a line of **53%** — at the line, so it skips. The input is the
contested part: *"of this week's 53 shared point(s): builder 23 (43%), desks 5
(9%), both 0 (0%), **NOT THIS PROJECT 25 (47%)**."* Nearly half the meter the
builder is gated on is a co-tenant's spend. That is `D40`'s subject, it is
`CONDUCT-DESK` and **12 days stale**, and it is unfloorable by design.

## RANK 4 — the builder is awake and the frontier has not moved in four days

- **Iterations in the last 24 h: ONE** (`06:07:09`–`06:14:56`, `rc=0`). The
  other 23 slots were `PACE-SKIP`, not failures: `0 failed slots`.
- **PASS delta: ZERO.** Every logged iteration from 2026-10-19 to now reads
  `107 -> 107 demonstrated`. That is 20+ consecutive `rc=0` iterations with no
  movement, while the registry grew **257 → 258** (`LC.08` registered
  `da9ece0`). **The ladder is getting longer and the demonstrated count is
  standing still.**
- Ledger: **160 rows / 107 PASS / 35 FAIL / 17 VOID / 1 BLOCKED** against **258**
  registered — **98 registered specs have no row at all**.
- The work that did happen was real and was instrument work: `LC.08` registered
  and twice strengthened, `T0.17` P10c', the venue comparator, the metric-reader
  repair, the `t306` null-walker probe executed and harvested. **None of it
  advances a `GOAL.md` capability.** `LC.08` is the nearest thing — it can
  indict 17.3% of the seated learning core as dead weight — and it is registered
  and **unbought** (routed by the Review this morning, `872dbc9`).

## RANK 5 — six `CONDUCT-DESK` entries are stale, oldest 29 days

`decisions --check` prints, with the instrument's own words — *"desk-executable,
not the owner's — execute it, report it, do not ask. Listed so a stale conduct
entry cannot silently self-approve"*:

| id | due | stale |
|---|---|---|
| `D33` | 2026-09-23 | **29 d** |
| `D35` | 2026-09-24 | **28 d** |
| `D38` | 2026-10-04 | **18 d** |
| `D40` | 2026-10-10 | **12 d** |
| `D39` | 2026-10-15 | **7 d** |
| `D46` | 2026-10-17 | **5 d** |

Twenty-nine days of being listed is what "cannot silently self-approve"
currently buys. I am not manufacturing a new ask for these — they are already
filed, already classed, and the instrument already names them every run. **Rank
1 is why this matters more than slowness:** these entries are argued on quoted
numbers, and nothing re-derives one.

---

## Section-by-section, including the clean results

**1. Integrity of the ledger — NO FINDINGS, and this is a real result.** All 160
rows carry a `commit`. Every one resolves in git except two `+dirty` markers —
`PL.02` (`7ffd3c8+dirty`, **VOID**) and `T6.03` (`03e68bc+dirty`, **BLOCKED**) —
and both base shas exist. **No PASS rests on a vanished commit, and zero PASS
rows lack a commit field.**

**2. Thresholds and controls over 7 days — NO FINDINGS. Re-derived, not
inherited.** The sweep over `registry.py`, `registry_expansion.py` and
`tests/`: no numeric threshold moved in the loosening direction, no control
deleted or weakened, no `_check` gained an `or`, no seed count reduced, no
assertion removed. Two commits strengthen: `87fc826` struck the refuted
teacher-caps-twin mechanism from `LG.03`'s live comments, and `b6069dc` ported
`T2.14`'s triviality floor into `T2.05`.

The one commit that *could* have been a silent loosening is `31a3fd0`, and I
checked it at source rather than reading its message. It replaced a live
cross-spec ledger read with a pinned `SEAT_RECORDED` constant. **I verified all
five pins against `LC.03`'s live row (attempt 3, `ran_at 2026-08-23T21:11:17`)
and every one reproduces to the byte.** The pin is fail-safe in the only drift
direction that exists — the sole event that could move the row regenerates the
artifacts, which then mismatch and VOID. **And its author disclosed his own
`T0.22` FAILs (`a23`/`a24`) in the commit message rather than hiding them.**
That is the conduct this organ exists to reward, and it should be said as
loudly as a finding.

**3. Drift from the goal.** The day's work serves the honesty clause
(*"protects the honesty of watching what happens"*) and not the capability
target. Against `GOAL.md`, `coverage` reads **0 commitments with no declared
spec** and **3 CLAIM-DEAD** — `smell`, `shelter/building`, `thermal (kills)` —
each with every claim spec PARKED or FORECLOSED. **I am deliberately not
re-asking this.** `D42` fired on 2026-10-19 (default **(iv) HOLD**), and its
firing already recorded the worse-than-stated price in writing: those three now
have no live claim spec, no open queue row and no open register entry, and
`coverage`'s `claim_dead = 3` is the one organ still saying it out loud. The
owning row was DECLINED on 10-17 by its own armed stop-rule. Every step was
legal; the hole is unchanged. `NO-LIVE-PATH` reads **6** commitments/seats with
no live path, **measure-only by design**. Fourteen commitments have live claim
specs with nothing passing — curiosity 2/12 passing, fast/slow **0/8**, sleep
**0/5**, hunger/thirst **0/6**, one-brain/unison **1/28**.

**4. Builder alive and productive** — Rank 3 and Rank 4.

**5. Compute honesty — a finding about the REPORTING, not the spending.** The
spending was recorded honestly and I want that on the record: `d212dd8` and
`bf409f8` committed both 10-20 GPU failures **as found**, both overruns are in
`gpu_budget.json` (`0.9909` vs est `0.7`; `1.0043` vs est `0.7`), and `77025f3`
**reattached** the recovered kernel rather than resubmitting ("no new GPU
spend"). That is good conduct under a bad outcome.

The outcome: `2026-W42` drew **2.078 h**, of which **1.995 h (96%) was failed
jobs** — `colab_failed 0.9909` + `kaggle_failed 1.0043` against `kaggle 0.0828`
succeeded. Both target specs (`T1.02`, `T1.09`) then passed on CPU with
`backend None, gpu None`, so no ledger row shows the GPU spend.

**The reporting defect:** `docs/PROGRESS.md` `FOR THE BUILDER` **item 0** and
`FOR THE OWNER` **item 4** both state `2026-W42` holds `30.0` free hours,
`0.0` drawn — and the page's own refresh banner explicitly marks item 0
**"FRESH, measured 2026-10-20"**, the marker that tells a reader *trust this
one*. The jobs landed 10-20 at 13:12 and 14:13, after that measurement. The
order built on it (*"do not let a third week close at 0.0 in silence"*) now has
a false premise, and nothing re-derives it. **Not re-routed**: the 158th audit
already routed `gpu-charge-ok-flag-is-frozen-by-the-dedupe-guard` (OPEN, DUE
2026-10-26), whose provenance line carries the `1.9952 h` waste figure.

**6. Stuck decisions** — Rank 1 and Rank 5. `decisions --check` exits **0**;
ratchet clean (`0/0` undeclared, unrouted-owner-ask, vanished-owner-ask,
default-action-expired, firing-diff); firing transcriptions **34 of 34**.
**Nothing was armed this audit because `UNDECLARED` reads 0** — recorded
rather than manufactured. `D44` and `D51` remain armed and intact after my
append (verified).

**7. Bakeoff hygiene — NO NEW FINDINGS.** `champions --check` exits **0**, every
class at baseline: `0/0` phantom arena, `2/3` unfalsifiable (ASR, Speaker ID),
`4/4` unwinnable, `2/2` unverified verdicts (`Learning core` `LC.03=VOID`,
`World` no deciding run named), `3/3` trigger debt, `1/1` kindless discharge
(`LF.02`). Every seat declares what would unseat it. The standing scandal is
unchanged and is not re-routed: the **World** seat is held **BY VERDICT** — the
file's strongest marking — with **no deciding run named** and **no `TRIGGER:`
declared**.

**8. The honest summary.** **We are not closer to a curious humanoid than we
were yesterday; we are closer to a measuring apparatus that can be trusted.**
That is not nothing — this morning I could verify 160 ledger rows, five pinned
constants and six days of threshold history and find no dishonesty, and a
project that cannot trust its green ticks cannot learn anything. But `107`
demonstrated has not moved in four days while the registry grew by one, 98
registered specs have never been run, fast/slow has **0 of 8** passing and
one-brain/unison **1 of 28**, and three of the owner's own survival directives
— *"too cold kills him"*, *"he builds a shelter"*, smell — are formally HELD
claim-dead with no row, no clock and no instrument that can escalate them.
**The thing that most needs saying: today's three findings are all the same
shape.** A number that was true once is quoted later as if it were measured now
— `D47`'s drain, `PROGRESS.md`'s `0.0` GPU-hours, `dark_slots`' zero. The
ladder is honest about experiments and is not yet honest about its own
instrument readings, and that is the gap through which a desk can work hard all
week on a premise that stopped being true on Tuesday.

---

## 9. FOR THE BUILDER

**0. Nothing in this section is urgent compared to a PASS.** The frontier has
not moved in four days. If a slot can be spent on `T2.05`'s ordered port (DUE
10-25, zero GPU, bill 0 certificates) or on `LC.08`'s purchase, spend it there
and leave every item below for the regate lane.

**1. Give the register a `T0.22`-shaped reader for quoted instrument readings.**
`T0.22` already forbids this exact shape for specs (*"A number borrowed from
another spec's entry cannot be stale"*). `docs/DECISIONS_NEEDED.md` has no
equivalent, and Rank 1 is the cost. Minimum viable: for any entry body or
`DECIDE:` block quoting a named instrument's reading, re-derive it and flag
drift. **Ruling first, not code** — routed as
`register-entries-quote-instrument-readings-that-nothing-re-derives`
(OPEN, DUE **2026-11-02**).

**2. Add a cumulative dark-slot reading beside the streak.** Three candidates
are on the row; this desk prefers `dark_slots_7d`. It is **monotone** — adds a
reading, moves no floor, cannot make an existing number look better. Routed as
`dark-slot-ratchet-is-a-streak-so-an-ENDED-blackout-reads-zero`
(OPEN, DUE **2026-11-04**).

**3. Print `DUE TODAY — n live row(s)` beside the OVERDUE class.** Not a gate —
a row due today is legal and a gate would forbid a legal move. Routed as
`overdue-reconstitutes-overnight-from-rows-the-desk-could-see`
(OPEN, DUE **2026-10-31**).

**4. BILL DISCLOSED UNPAID.** My `REVIEW_QUEUE.md` and `DECISIONS_NEEDED.md`
edits stale `T0.21`/`T0.28`/`T0.31`. Priced with `scripts/regate.py --list`
**before** the first edit: **3 certificates / 327 s / ~0.09 CPU-h** — and all
three were **already stale** when this audit opened (the Review's own 06:38
act), so **my edits add zero incremental certificates**. Running specs is
outside this organ's remit; the regate stale-first lane is the expected payer
and the next builder slot the fallback. The tree was clean at pricing time
(`regate` read *"would stamp 0dd90eb (clean tree)"*).

**5. DISCLOSURE — the cross-organ doc race fired on me, twice.** All three of
my rows are committed and durable, but **none of them under my own commit**:
`5404cc3` swept rows 1–2 and `3ed8d0d` swept row 3, because the Review staged
`REVIEW_QUEUE.md` by name while my appends were uncommitted. The rows are
therefore attributed to *"review DAILY 10-22 act 4/5"*, which is not what they
are. This is the live `cross-organ-doc-race-voids-certificates` class. Anyone
auditing the provenance of these three rows should read this paragraph, not the
commit messages. **I also watched the Review introduce and then self-repair a
`MALFORMED` row in that window** (`5404cc3` put a fifth pipe on
`priority-block-orders-…`; `99b2319` fixed it) — and note what the instrument
did well there: `review_queue_violation_forms` moved
`{'OVERDUE': 16}` → `{'MALFORMED': 1, 'OVERDUE': 7}`, so the per-form counter
caught a violation converting into a different violation. The OVERDUE class
briefly fell by one for a reason that was not a disposal, and the total held at
8. **That is `T0.31` working exactly as specified.**

## 10. FOR THE OWNER

**1. Nothing new is being put on your desk this morning, deliberately.** Six
`CONDUCT-DESK` entries are already stale by up to 29 days (Rank 5) and a
seventh ask would make that worse, not better. Rank 1 was mine to correct and I
corrected it; Ranks 2–4 are routed with dates.

**2. The one thing that is genuinely yours is unchanged and I will not re-ask
it.** `D42` fired on 2026-10-19 and HELD: `smell`, `shelter/building` and
`thermal ("too cold kills him")` stay claim-dead. Naming **one** of them as the
one that matters most remains yours, in one sentence, at any time — and the
firing deliberately did not pick for you, because choosing among your own
constitutional commitments would be a desk writing `GOAL.md`. **What I want you
to see is the shape of it now:** the question has no open row, no open register
entry and no clock. It survives only as the number `claim_dead = 3`, which sits
at its own floor and therefore never goes red. Three of your 2026-08-09
survival directives are unfalsifiable and the system has stopped being able to
raise its hand about it.

**3. The second genuinely-yours item, also unchanged: two weeks of free GPU
hours have now expired without anyone being able to decide whether to spend
them** — and this week's reading is worse than a third zero, which is the
correction. `2026-W41` closed at `0.0`. `2026-W42` is **not** `0.0` as
`PROGRESS.md` still says: it drew **2.078 h**, of which **1.995 h (96%) was
failed jobs**. So the pattern is no longer "the hours expire unused"; it is
"the only hours we drew, we wasted, and the page that orders the builder not to
let a third week close at zero is working from a number that stopped being true
on 10-20 at 13:12." I do **not** recommend buying the remaining hours: the two
GPU specs that could absorb them (`T2.03`, `T2.14`) both need `T1.08`, which is
**FAIL at attempt 5** with a sixth attempt prohibited. **The asset is not the
quota. It is a decision about `T1.08`, and that decision is 12 days stale on
this desk's own docket as `D40`.**
