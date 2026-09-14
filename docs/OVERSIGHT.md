# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **96th audit — 2026-09-14, 06:38–07:0x UTC.** Read at HEAD `a75b5c5`, clean
> tree. The builder is awake and between slots (06:07 slot ended `rc=0` at
> 06:24). 24 iterations in the 24 h, 23 `rc=0`, one with no end line (09:07
> 09-13, the `rc=124` the 95th audit recorded). `demonstrated` 107 → **108**
> over the 24 h against a registry that grew 246 → **249**, and **108 → 108
> across the last thirteen consecutive iterations**. The Sunday FULL ran
> 09-13 06:37; `docs/PROGRESS.md` still carries it; the next Review sitting is
> a DAILY, today.

> **TWO CORRECTIONS, both landing during the sitting and both against me.**
>
> **(a) RANK 1's headline fact expired 52 minutes after I measured it.** The
> Review's DAILY sat at 06:45–06:50, consumed the field watch (`f34d366`) and
> **routed the A4 finding itself** — `a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere`,
> OPEN, **DUE 2026-09-18** (`0ae7c71`), the same date my B2 named, reached
> independently. The finding is **no longer homeless**; B2 is discharged before
> it was read. RANK 1 is corrected in place below and its *reader* half stands
> unchanged — see the corrected text for what survives and why I did not retract
> the whole rank.
>
> **(b) This audit's three files were swept into another organ's commit.** My
> `OVERSIGHT.md`, `D29` and the lesson were staged at 06:49:30 and committed at
> 06:49:48 inside **`6698fc8`**, whose message is about a blast-radius contract
> and mentions none of them. Nothing was lost and nothing was corrupted — I
> diffed all three. But the git log of this page now attributes the 96th audit
> to a Review commit, so **`cross-organ-doc-race-voids-certificates` has a fresh
> instance, produced on the overseer, while that row is itself OVERDUE** (promised
> 2026-09-13, 1 d). It is dated evidence for a live row and it is recorded here
> rather than filed as a new finding.

## VERDICT: DRIFTING

**The clean part first, and it is most of the audit. Every number below I
recomputed; none is quoted from another organ's report.**

- **Section 1, integrity. No finding.** 108 PASS rows. **Zero** without an
  implementation in `experiments/tests/`. **Zero** without a declared
  `control`. **322** distinct commits referenced across every `history` entry,
  **zero dangling** after stripping `+dirty`. `run stale` lists 11 stale claims
  + 1 pre-`impl_sha` — and **not one of the twelve is a PASS**: they are 5 FAIL,
  6 VOID and 1 VOID. No standing certificate rests on code that has moved.
- **Section 2, thresholds. No finding.** 43 commits touched
  `registry.py` / `registry_expansion.py` / `tests/` in 7 days. Every numeric
  move was in the **harder** direction: `MAX_SPREAD_RATIO` 6.0 and
  `MAX_HELDOUT_CV_PCT` 7.0 as NEW conjuncts, `T6.03`'s five additive conjuncts,
  `MIN_DISTRACTOR_EVAL` 30 → 59 and `N_DISTRACTOR` 60 → 130 on three ME specs.
  Two apparent hits checked by hand and both are clean: `MIN_PARA_MARGIN
  0.10 → 0.05` is a **synthetic diff inside `decisions.py`'s firing-diff
  fixture** (live `t2_10` still reads 0.10), and `D1.0`'s new
  `or raw paired gain < LEARN_MARGIN` is an **added disjunct in a VOID
  condition**, i.e. stricter, with `MIN_LEARN_SIGMA` unmoved at 3.0. **Nothing
  loosened.**
- **`coverage`'s highest-priority check is GREEN: 0 commitments with NO declared
  spec.** EXIT 2 is `claim_dead` 4, the FILL-HELD class (`HR.1` ← `D19`) and the
  four no-path cost classes — all owned, all routed, all unchanged.
- **Section 5, compute. No waste this week.** `2026-W37` (Sunday-start, opened
  09-13): **0.82 h of 30 spent, 29.18 h free, 0 overruns.** Whole-history
  overrun count is still zero. `gpu_hours_no_verdict` TOTAL 48.42 h unchanged,
  `gpu_unattributed_jobs` 21 **AT** floor. Nothing was wasted because nothing was
  spent — see RANK 4 for why that is not a compliment.
- **Section 6, decisions. Ratchet ok.** `0/10 undeclared, 0/3 unrouted-owner-ask,
  0/0 vanished-owner-ask, 0/0 default-action-expired, 0/0 firing-diff`.
  `MEANS-ESCALATED` empty, `OVERDUE` empty. Four armed: `D19` (09-14),
  `D20` (09-18), `D27` (09-20), `D28` (09-21). **`D19` is due today and is
  fireable from 09-15 00:00**, not today — `decisions.py` marks overdue at
  `(today − decide_by).days > 0`, and the builder caught the steering page being
  looser than the tool on two separate slots. Nothing to arm; I added one entry
  (`D29`, below), which is an addition and not an arming.
- **Section 7, bakeoff hygiene. No finding.** `D25` fired 09-14 00:10 with the
  required wording, a reversal path, 14 new test assertions, and **one declared
  deviation that is measured from git rather than asserted** — the default's
  own letter required a conjunct that is FALSE of the very run the decision
  cites, so taken literally the branch could never have fired on its own scar.
  Declaring that is the correct move and it was declared. No VOID treated as a
  verdict, no winner inside a noise margin, no owner-decision acted on unrecorded.
- **`champions --check` rc=0**, every class at or inside its floor, 0/0
  UNDECLARED seats.

**Why DRIFTING, in one sentence:** at 05:57 this morning the project's only
outward-looking organ found that the **mandatory silent-failure kill-switch under
the Learning-core seat was never implemented** — and six hours later that finding
has **no queue row, no clock, no decision entry, and no instrument in this
repository that can read the file it lives in**, while the *smaller* finding in
the same sweep was diagnosed, repaired and re-bought inside 21 minutes.

**Why not INTEGRITY RISK:** no capability claim on this ledger is false, nothing
loosened, every hard number I recomputed agreed with the record, no PASS rests on
changed code, and the two findings that do bite (RANK 1 and RANK 2) are both
*understatements of evidence*, not overstatements of capability.

---

## RANK 1 — the Learning-core seat's declared kill-switch was never built, and the finding is homeless

`docs/research/LEARNING_CORE.md:1935`, verbatim:

> *"Collapse is the failure mode and it is silent, so A4 carries a **mandatory
> diagnostic: effective rank and per-dimension variance of the latent must be
> reported every 1,000 decisions**, and a collapse (rank below a pre-registered
> floor) is `Status.VOID` for A4, not a good loss curve."*

**I ran the grep myself rather than take the field watch's word:**

```
$ grep -rn "effective rank\|effective_rank\|RankMe\|per-dimension variance" --include=*.py experiments/
experiments/registry.py:892             "...per-layer effective rank every cycle. "
experiments/registry_expansion.py:4242  "...fraction and effective rank stay near their early-life "
```

**Both hits are prose inside other specs' `hypothesis` strings.** One belongs to
a plasticity spec, one to the sleep-downscaling spec. Neither is an
implementation. `LC.03`'s committed row records 50 metrics for `wm-latent` and
not one is a rank or a per-dimension variance, on any of five arms.

**Why this reaches the architecture and not just a document.** `A4` =
`wm-latent` **holds the Learning-core seat**, seated 2026-09-01 by `D10`'s armed
default, marked **BY VERDICT** — `CHAMPIONS.md`'s strongest marking. The seat
decides what Jack's brain *is*. Its declared guard against the one failure mode
the document calls *silent* was never computable, so the seating run could not
have detected a collapsed latent even if it had one. `SYSTEM.md`'s own rule
covers this exactly: *"A governing document that names an enforcement is making
a capability claim, and it is bound by law 1 like any other."*

**And it can never now be measured.** Three independent closures, each already
on the record: the trained `A4` weights **do not exist on disk** (field watch
wk6, cited in `CHAMPIONS.md:60`); `LC.03` v2 is **VOID-FORECLOSED** — no v3, no
envelope growth, no re-roll, pre-registered; `LC.07`, the seat's only live arena,
is **VENUE-UNAFFORDABLE** (`D24`'s armed default, ~526 wall-hours ≈ 17.5 weeks of
every GPU hour this project has). **This is why it is not a rule-3 violation to
put it on the owner's desk: no bakeoff this project can run would settle it.**
Routed as **`D29`** with that reasoning attached.

### The part that is mine to report, and it is the worse half

```
$ grep -rn "effective rank\|§5.4\|collapse diagnostic" docs/REVIEW_QUEUE.md docs/DECISIONS_NEEDED.md docs/PROGRESS.md scripts/ladder_prompt.md
  (no hits)

$ grep -rn "FIELD_WATCH" --include=*.py experiments/ scripts/
  (no hits)
```

**Nothing in this repository reads `docs/FIELD_WATCH.md`.** Not `run status`,
not `coverage`, not `decisions`, not `review_queue`, not the seal. The sweep's
findings are consumed by one hand-written line in the Review's Part 2.5
(*"Field watch — nothing to consume"*), which is a habit, not an instrument.

The builder's 06:07 slot said so in the open and was not hiding it: *"The §6
A4-diagnostic finding is a three-way decision fork the FULL will read directly
from the field watch; **I routed nothing there.**"*

**CORRECTED AT 06:50 — the finding was routed while I was writing this, and I
was wrong about the outcome.** I predicted it would wait for Sunday's FULL and
not be read there either. Instead the Review's **DAILY** consumed the field
watch in full (`f34d366`) and routed
`a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere` at 06:49
(`0ae7c71`), OPEN, **DUE 2026-09-18** from `next_free_due` — the same date my B2
was about to name. It reproduced both greps independently, recorded a leaning of
*"(i)+(iii) together, never (iii) alone"*, and wrote the sentence that makes the
routing honest: *"an amendment that records its own scar is a repair; one that
quietly matches text to code is how a capability claim disappears."* It also
indicted itself in the same commit for the identical failure one week earlier.
**That is the desk working well and my forecast was simply wrong.**

**What survives the correction, and it is the part I would not retract.** The
finding was discharged **because a Review sitting happened to run within the
hour and happened to read a page nothing obliges it to read** — Part 2.5's
*"Field watch — nothing to consume"* is a hand-written habit. `grep -rn
"FIELD_WATCH" --include=*.py experiments/ scripts/` still returns **zero**. The
habit held this morning; it is not a mechanism, it is not counted anywhere, and
nothing would have gone red had the sitting been the one that dies at its wall
clock. **B1 is unchanged and is now better-evidenced, not worse:** the gap
between a finding landing and an instrument being able to see it was **52
minutes of luck**, and the measurement of that gap is exactly what an
`UNROUTED-FIELD-FINDING` counter would have made visible instead of leaving to a
forecast — mine, which was wrong.

**This is `D15` one organ over.** `decisions.py` was taught to print
`UNROUTED-OWNER-ASK`/`VANISHED-OWNER-ASK` after the 09-03 loss, precisely because
an ask living only on a page dies there. `REVIEW_QUEUE.md` got a reader on 08-31
under my own B4 for the same reason. The field watch is the third page in this
pattern and the only one still unread by any tool — and it is the one that looks
*outward*, so what it finds cannot be re-derived from the ledger by anybody else.
**B1 below.**

---

## RANK 2 — two standing PASS certificates carry a 0.95 abstention bar their denominators cannot certify

The same sweep's §6b, and the numbers are mine from `ledger.json` rather than
from the table:

| spec | `distractor_evaluated` | a perfect 1.0 certifies `a_L = 0.05^(1/m)` | the conjunct's bar | status |
|---|---|---|---|---|
| `ME.9` | **15.0** | **0.819** | 0.95 | **PASS, unrepaired** |
| `ME.10` | **36.0** | **0.920** | 0.95 | **PASS, unrepaired** |
| `ME.3` | 87.7 | 0.966 | 0.95 | PASS, repaired 06:24 |
| `ME.1` | 94.7 | 0.969 | 0.95 | PASS, repaired 06:24 |
| `ME.5` | 110.3–130 | ≥ 0.973 | 0.95 | PASS, repaired 06:24 |

`m ≥ 59` is the γ=0.05 minimum (`0.05^(1/59)` = 0.9505). **`ME.9` reads
`distractor_abstention = 1.0` at exactly 15 negatives, which is statistically
compatible with a true abstention rate of 0.819** — 13 points below the bar the
conjunct names. It is not a false claim and nobody cheated: the conjunct is
satisfied as written. It is a **conjunct that cannot bear the weight its own
number puts on it**, which is section 1's "a control that was never run" in its
softer form.

**Why `ME.9` in particular.** GOAL.md names it by id — *"ME.9 (attributed recall
of heard/said/did)"* — and `CHAMPIONS.md` names it as one of three arena members
for the **Person model** seat, the only seat any anatomy audit has created that
was contestable on the day it was created. That contestability is priced partly
on `ME.9`'s PASS.

**Credit where it is owed, because the builder's handling was good.** The gap was
found at 05:57, three of five were repaired and re-bought PASS by 06:24 with the
0.95 bar untouched in both directions and the `N_DISTRACTOR` raise verified
offline first, the staleness bill was paid (`ME.11.A` re-bought), and the two it
could not fix mechanically were recorded as an evidence addendum on
`me1-similarity-floor-never-abstains` — **which is DUE TODAY**. That is the
right disposition.

**What is missing is a reader, again.** `ME.9` and `ME.10` now sit on the board
rendering `[PASS]` **indistinguishably from the three that were repaired**.
Nothing in `run status`, `coverage` or the ledger prints that two certificates
carry a bar their denominator cannot certify. **B3 below**, and note it is the
*same shape* as RANK 1: the finding exists, the arithmetic is settled, and no
instrument carries it.

---

## RANK 3 — the strengthening direction has a reachability discipline, it was applied, and it was not strong enough

**I want to be exact here, because the lazy version of this finding is wrong.**
The 92nd audit's B3 discipline *was* run before the Review's three new conjuncts
were armed (`3d357c4`, committed deliberately **before** `T1.07`'s result landed
so the certificate would not stamp `+dirty`). It cleared two and **foreclosed one
— an audit's own order**. That is the ratchet working.

It was still insufficient, in one diagnosable way:

```
heldout_cv_pct <= 7.0   recorded 5.717, n=1, floor > 0 by the control.
```

A reachability check on **one observation of a sample statistic** says nothing
about that statistic's sampling spread. `T1.08` then read **40.006** and went
PASS → FAIL, and the builder priced the bar with zero GPU at 02:07 today: a
3-seed CV drawn from a pipeline whose *true* CV is exactly the 5.717 the bar was
set from lands in **[0.92%, 11.00%]** 95% of the time and **fails a 7.0 bar
22.6% of the time**; the whole discordance is a **4.0%** event under *nothing
changed at all*; and code drift was eliminated by hand as the third explanation.

**The cost of that one bar, measured:** `T1.08` is now the largest blocker on
the board — `frees 3 (blocks 45)` — it took `UNREACHABLE` 94 → 97, it put
`D1.0` and `T2.01`'s 38-spec unblock behind it, it left **2 standing PASS
certificates UNBACKED** (`T2.03`, `T2.14` — a legal state, but a re-buy is not
free), and it is why **29.18 free GPU-hours have no legal buyer** with the week
expiring 09-20.

**The sibling is still unpriced, and the builder flagged it rather than
quietly carrying the number across.** `T1.07`'s `spread_ratio ≤ 6.0` was armed
in the *same commit*, by the *same* transported-ratio rule (1.217× → 1.224×),
and it is **PASS**. Its statistic is a max/min over LR arms, not a std over
seeds, so the 22.6% must not be carried onto it — pricing it needs seed noise
nobody has measured. That refusal is correct and I am not asking for it to be
overridden.

**And the thinnest margin in `T1.07` is the CONTROL's, not the claim's:**
`absurd_advantage < 1.15` sits at ×1.255 headroom against a quantity **measured
to move ×99.59 between venues**. A claim conjunct failing gives a legible red
row; a control conjunct failing gives a **vacuous PASS**. Both are on
`t108-noise-floor-is-quoted-by-nobody` as addenda (e′)/(e″), **DUE 09-16** —
so this one is owned and clocked, and I checked that before writing it up. The
builder also scoped the backward sweep honestly and **closed it as not-a-unit**
(840 pairs, 92.1% frozen to the recorded digit, exactly one ≥×10 mover, already
found by hand), routing the forward rule instead. No finding against the builder
here; the finding is that **no instrument counts a control's margin**, and the
repair is forward-only by the builder's own measurement.

---

## RANK 4 — the builder is healthy, productive, and starved; the constraint is entirely downstream

**Section 4, and the numbers do not say what the commit count says.**

| | reading |
|---|---|
| iterations, 24 h | **24** (23 `rc=0`, 1 with no end line at 09:07 09-13) |
| commits, 24 h | **107** |
| `demonstrated`, 24 h | 107 → **108** (+1), registry 246 → **249** |
| `demonstrated`, last 13 consecutive iterations (18:07 → 06:24) | **108 → 108** |
| settle runs, 7 d | **74** — 5 first-ever verdict, **65 re-buy**, 4 status change |
| instrument-coupled settles | **46 of 74 (62%)** — our own tool edits staled the certificate |
| GPU spent, 24 h | **0.00 h**; W37 free **29.18 h**, expires 09-20 |

**Every first-ever verdict in the window landed in one 3½-hour block**
(09-13 14:22 → 17:25): `T0.36` PASS, `SO.10` FAIL, `LG.13` PASS, with `LG.12`
FAIL and `PL.02` VOID just before. **Two of those are architecture seat races**,
which is the 08-24 owner ruling actually being executed: `LG.13` took the
Language-routing seat BY VERDICT the day after the seat was created, and `SO.10`
raced the Person-model seat and **its own winner could not hold it**, leaving the
seat VACANT BY MEASUREMENT. A seat race whose winner fails is the ladder standard
applied to the brain's own organisation, and it is the best work in the window.

**In the thirteen iterations since 18:07, not one spec about Jack was run.**
Every unit was an instrument, a legality reader, or an arithmetic pricing
exercise on an existing bar. **This is not idleness and I checked rather than
assumed.** The builder verified an empty board three independent ways on four
separate slots (`run next` 0 fresh; `coverage`'s only fillable class FILL-HELD by
`D19`; the steering reader flagging `D1.0` as an order the runner would refuse),
refused `fable` correctly on the model floor for 22 consecutive slots, read
`week:all models` as the gate every time, declined to manufacture a GPU dispatch
into 29.18 idle hours four times, and corrected two steering pages that were
looser than the tools. It also shipped a false positive in its own new reader and
**caught it with arithmetic** (2 of 10 → 1 of 10, keeping the live false positive
executable in the fixture so deleting the rule turns it red).

**The constraint is one layer down.** The exits from the empty board are `T1.08`
(45 specs, fixture ruling DUE 09-16), `T2.01` (38 specs, behind `T1.08`), and the
four CLAIM-DEAD commitments' redesigns — **every one of them owed by a desk whose
drain reads UNBOUNDED and which broke 13 dated promises at midnight**. That is
`D28`, routed by this organ six hours ago; cited here, not re-asked.

---

## RANK 5 — the queue, and the one number that is not noise

`run review-queue` **EXIT 2 — 13 OVERDUE**, all promised 2026-09-13, all 1 day
old. 33 OPEN / 3 HELD / 13 DISPOSITIONED of 59 routed; oldest live 21 d; consumer
last ran 09-13. `review_queue_violations` **0 → 13**, the first violations this
queue has carried in its recorded life, and **nothing has recorded or justified
the movement.**

`review_queue_net_arrivals` read `MOVED −3 (clock −3, act +0)` — and the
decomposition the builder shipped under the 95th's B1 is doing its job: it
correctly attributes the whole move to the sliding trailing window and says *"no
act, nothing to investigate, and no commit can justify recording it."* The 95th
audit called that counter's calendar-driven drift its headline. **One day of the
split existing has already converted it from a mystery into a labelled zero.**
I am downgrading it accordingly — it is repaired, not merely explained.

`next_free_due` is **2026-09-18**. Re-dating 13 rows onto one cycle of measured
capacity 6 rebuilds the pile; **DECLINE remains honest and entirely unused — 0
of 59 rows have ever been declined.**

---

## Section 3 — drift from the goal

**No unit in the last day serves no GOAL.md sentence.** The two seat races serve
`SYSTEM.md`'s ARCHITECTURE-always-contested invariant and GOAL.md's *"this
project depends on research and testing at EVERY SINGLE STAGE"*; the ME
denominator repair serves *"really learning, not appearing to learn"*; the
pricing units serve *"every capability claimed only by an experiment that could
have failed"*. The instrument work is the honesty layer, which GOAL.md's opening
explicitly counts as in-scope (*"or protects the honesty of watching what happens
when the three meet"*).

**The converse is the uncomfortable half and it has not moved.**
`GENERALITY.md`: 14 barriers named, 4 registered, **0 RUN, 0 PASS** — byte-identical
to the 09-06 and 09-13 readings, and it has now received zero seconds of compute
for a third consecutive week. Four constitutional commitments remain **CLAIM-DEAD**
(smell, balance, thermal, shelter) and three champion seats **unwinnable**; the
`NO-LIVE-PATH` count is 7, unchanged. Curiosity has 12 specs and 2 PASS;
all-senses fusion (`one brain / unison`) has 27 specs and **1 PASS**;
learning-by-living (`death & retry`) has 6 specs and **0 PASS**. Those three are
named in my own prompt as the most likely to be quietly neglected, and they are
being quietly neglected — not by choice, but by being downstream of `T2.01`,
`NE.01` and a world that is the measured bottleneck on six instruments.

---

## Section 8 — the honest summary

**Yesterday afternoon, yes. Since 18:07, no, and the reason is structural.**

For three and a half hours on Sunday this project did the thing it says it does:
it created two architecture seats and then *raced them the same day*, and one of
the races **killed its own winner** — `SO.10` FAIL, seat stays VACANT BY
MEASUREMENT. A system that seats champions by argument would have taken the win.
That is the ladder-and-apple standard pointed at the brain's own organisation,
and it is real progress toward the goal rather than toward a longer list of
green ticks.

Then thirteen straight iterations moved `demonstrated` not at all, and every one
of them was right to. The board is genuinely empty; every exit runs through a
desk that cannot pay; 29.18 GPU-hours will expire unspent on Saturday. The
builder spent that time making the instruments honest — and, to its real credit,
spent two of those slots proving that a unit it had been handed was **not worth
doing** (the backward control-margin sweep, closed as not-a-unit on measured
evidence) rather than doing it to look busy.

**What actually worries me is RANK 1, and it is a shape and not an incident.**
The field watch produced two findings this morning. The one with an arithmetic
formula and a fixture constant behind it was fixed in 21 minutes. The one about
whether the seat that decides what Jack's brain *is* was won with its
kill-switch disarmed got nothing — no row, no clock, no reader — because no
instrument in this repository can print it. **This project allocates its
attention by what a tool can print, and it has become very good at that.** The
consequence is that findings arrive in order of how mechanisable they are, not
how much they matter, and the least mechanisable finding of the last four months
landed this morning and is currently scheduled to be read by nobody. That is
the drift. It is not dishonesty and nobody hid anything — the builder said out
loud that it routed nothing there. It is that we built a nervous system with one
sense organ still unwired.

---

## FOR THE BUILDER

Ordered. None of these is a GPU unit and none needs the desk's permission.

1. **`docs/FIELD_WATCH.md` has no reader — give it one. This is the unit.**
   `grep -rn "FIELD_WATCH" --include=*.py experiments/ scripts/` returns
   **zero hits**. Build the reading in the exact idiom of `decisions.py`'s
   `UNROUTED-OWNER-ASK` (which exists because of `D15`) and of `review_queue`'s
   own reader (which exists because of my B4, 08-31): resolve the current
   sweep's dated findings-sections against `REVIEW_QUEUE.md` rows and
   `DECISIONS_NEEDED.md` entries, and print **`UNROUTED-FIELD-FINDING`** in
   `run status` alongside the other ratchet counters. **Report-only and
   unfloored until its false-positive rate is measured and written down** —
   that is `D27`'s reasoning and it applies here unchanged; do not floor a
   counter whose parse you have not measured. Ship it with the **live** finding
   (`§6`, A4's diagnostic) executable in the fixture, so deleting the rule turns
   the test red — the same trick you used on the steering reader this morning.
   Count every class, per `T0.31`: a reader that counts one class pays a repair
   that lowers its own number.
2. **~~Route `§6` today, with a clock.~~ DISCHARGED 06:49 by the Review
   (`0ae7c71`), before this page was committed** — same date (09-18), reached
   independently. One thing remains and it is small: the Review's row lives in
   `REVIEW_QUEUE.md`, so **`champions.py` and `CHAMPIONS.md` still cannot see
   it.** When the row is disposed, record the fact **on the Learning-core seat
   cell**, beside the single-arm caveat and the `VENUE-UNAFFORDABLE` label it
   already carries. **Do not change the seat's `HELD:` marking** — downgrading
   `BY VERDICT` would move `champions --check`'s UNVERIFIED-VERDICTS count 2 → 1
   and shrink a ratchet by re-labelling instead of by repair, the `ARENA-MISSING`
   anti-pattern wearing a different hat. The marking question is `D29`'s and it
   is the owner's; the Review's stated leaning ((i)+(iii)) does not touch it, so
   the two are not duplicates.
3. **Make `ME.9` and `ME.10` distinguishable from the three you repaired.**
   They render `[PASS]` today exactly as `ME.1`/`ME.3`/`ME.5` do, and nothing
   prints that their `distractor_abstention = 1.0` sits on m = 15 and m = 36,
   certifying **0.819** and **0.920** against a conjunct that names 0.95. The
   cheap half is a `run amend --doc-only` on both docstrings recording the
   certified level beside the bar — AST-proven prose-only, no certificate
   staled, which is the same lane you used on `T1.07` and `T1.08` this morning.
   The fixture redesigns are already owned by
   `me1-similarity-floor-never-abstains` (DUE today); do not pre-empt them.
   **The 0.95 bar does not move in either direction.**
4. **When you next touch `T1.07`, price `spread_ratio ≤ 6.0` the way you priced
   `T1.08`'s 7.0.** Not now, and not as a GPU unit — you were right that it needs
   seed noise nobody has measured and right to refuse to transport the 22.6%
   onto a max/min statistic. This is a note so the refusal does not become a
   silence: `T1.07` is the one PASS on this ladder standing on a bar armed by a
   transported ratio whose false-fail rate is unknown.
5. **`D19`'s NO-FETCH default fires from 2026-09-15 00:00**, with the required
   wording: *"the owner did not rule by 2026-09-14, so the pre-registered default
   fired."* You have correctly declined it twice today; tomorrow it is due.
   The steering pages still say "fires 09-14" and are looser than the tool —
   fix the page in the firing commit.

---

## FOR THE OWNER

**1. `D29` — NEW, routed today, `class: goal`, `decide_by` 2026-09-22.**
*The Learning-core seat — which decides what Jack's brain is — is held by
`wm-latent` under `CHAMPIONS.md`'s strongest marking, `BY VERDICT`. Its own
governing document declares a **mandatory** collapse diagnostic whose violation
is `Status.VOID` for that arm. The diagnostic is computed nowhere in this
repository, and it can never now be run on the seating evidence: the trained
weights are not on disk, `LC.03` is VOID-FORECLOSED by its own pre-registered
rule, and `LC.07` is VENUE-UNAFFORDABLE at ≈17.5 weeks of every GPU hour this
project has. Does a seat keep the file's strongest marking when its declared
silent-failure guard was never armed?*

My recommendation is the armed default: **(iii) RECORD THE DEBT, CHANGE NO
MARKING** — free, monotone, deletes nothing and moves no counter. I am telling
you plainly what I did *not* make the default and why: **(ii) correct the
document** would turn a visible unfulfilled guard into no guard at all, which is
the same move as deleting an arena reference to clear `ARENA-MISSING`; and
**(iv) downgrade the seat** would shrink `champions --check`'s UNVERIFIED-VERDICTS
ratchet by re-labelling rather than by repair. Neither may fire by silence.
**Rule 3 checked explicitly and stated on the entry**, because a means-fork on
your desk is the `D1` disease: the settling measurement is unavailable at any
price this project can pay, and I attached the three independent closures rather
than asserting it.

**2. `D28` is six hours old and due 09-21.** Cited, not re-asked. The 13 OVERDUE
rows are unchanged since it was written; the desk has not sat today.

**3. `D19` is due today and its armed default fires tomorrow.** NO FETCH. It
holds `HR.1`–`HR.4` and it is the only FILL-HELD cost class on the board. You can
still rule before 00:00.

**4. NO-DECISION — the honest read on the week's science, so you can match it to
the numbers yourself.** `demonstrated` moved +1 in 24 hours across 107 commits,
and 65 of the last 74 ledger settlements were re-buys of certificates our own
tool edits staled. That is not a builder failure — I checked its board-empty
claim four separate ways and it held every time. It is that **every exit from
the empty board is dated on a desk that broke 13 promises**, and 29.18 free
GPU-hours expire on Saturday with no legal buyer. `D28` is the lever.

**5. NO-DECISION — `D29` is not a duplicate of the Review's new row, and here is
the boundary so nobody has to guess.** The Review routed the A4 *fork* at 06:49
(`a4-...-computed-nowhere`, DUE 09-18): build the readout, and/or amend §5.4 to
record its own scar. `D29` asks the one question that fork does not touch and
that no desk may settle for you — **whether an architecture seat may hold
`CHAMPIONS.md`'s strongest marking while its declared guard was never armed.**
If you rule (iv), the Review's row inherits a marking change it currently has no
mandate for; if you rule (iii), its leaning and my default agree and the entry
closes cheaply. Either way `decide_by` 09-22 sits after the desk's 09-18, on
purpose, so you rule with its work in hand.

**6. NO-DECISION — liveness, nothing to rule on.** Builder alive, 24 iterations,
23 `rc=0`, last 06:24. Field watch fired on cadence this morning (wk7, 05:57) and
**declared its own degraded coverage rather than hiding it** — the arXiv API 429'd
on six attempts across two paths, so it ran none of the 40-entry enumerations
weeks 4–6 ran, and it put that in its coverage table instead of a caveat. It also
recorded a gap in its own method (four sweeps searched `cs.*` for a paper that
lives in `q-bio.NC`). That is an organ auditing itself correctly, and it is the
organ we have not wired to anything.
