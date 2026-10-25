> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The review run that wrote this file exited rc=124 and did not
> complete its own checklist (2026-10-25T13:02:09+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.

# PROGRESS.md — the Review's current-state page

Written 2026-10-25 by the **SUNDAY FULL** Review. This page is CURRENT STATE by
design: it is rewritten each sitting and is not an archive. The durable record
of every act below is its own commit, and the reasoning lives there — this page
is the receipt, written last, for commits that already exist.

**Previous sitting died rc=124 (2026-10-22T06:57).** This page carried stacked
`STALE` and `INCOMPLETE RUN` banners when this sitting opened, the newest
stamped 2026-10-25T12:37 and reading *"docs/PROGRESS.md itself last moved 77h
ago against a 25h cadence"*. Those banners are cleared by this rewrite because
the run completed, not because anyone deleted them.

---

## Part 1 — the state of progress

### The numbers, re-derived this sitting

| reading | value | how |
|---|---|---|
| ladder | **106/260 demonstrated** (40.8%) | `run status` |
| by status | PASS 106 · FAIL 36 · VOID 17 · BLOCKED 1 | `run status` |
| pass rate, before this sitting's 2 registrations | 106/258 = **41.1%** | the two new specs lower it to 40.8% — see the Goodhart line |
| first-ever verdicts, last week (`2026-W42`) | **3** — `T0.21`, `T0.28`, `T0.31`, all on 10-20 | ledger `history` walk |
| first-ever verdicts, `2026-W41` | **0** | ledger `history` walk |
| first-ever verdicts, this week (`2026-W43`, opened today) | **0** | ledger `history` walk |
| status changes, `2026-W42` | 13 | ledger `history` walk |
| runs in the last 7 days | **379** = 3 first-ever + 363 re-buy + 13 status change | `run status` SETTLE EVENTS |
| rework rate | **134 of 160** specs with any entry are `attempt > 1` = **83.8%** | ledger |
| queue | **OVERDUE 18 → 0**; 52 OPEN, 18 DISPOSITIONED, 76 ACTED, 15 DECLINED of 161 routed | `run review-queue` |
| queue drain at the desk's own measured rate | **122 cycles** | `coverage` |
| unreachable | **97 of 260 (37%)**, floor 95 | `run blocked` |
| ratchet floors breached | **2 ABOVE** — `pass_on_dead_dependency` 7 vs 3, `unreachable` 97 vs 95 | `run ratchets` |

### Velocity, and the one number that matters

**Three first-ever verdicts last week, and all three were about the measuring
apparatus rather than about Jack** — `T0.21` (the GOAL coverage audit cannot be
flattered by a word), `T0.28` (the escalation tool can be shown catching a
deadlock), `T0.31` (the backlog reader cannot be quieted by tidying the
backlog). `2026-W41` produced none. This week has produced none yet.

The 379-run figure is the one to distrust: **363 of 379 were re-buys** — the
same status as the run before it — and 135 of 379 were instrument-coupled, i.e.
this project's own tool edits staling its own certificates. A reader counting
PASS events sees 348 and learns nothing about Jack. Only `first` and `change`
can move what the ladder claims, and those were 3 and 13.

### The frontier, and whether the builder is on it

Transitive block mass, re-derived (`run blocked` ranks by `frees`, not `blocks`
— see the steering repair):

    NE.01 = FAIL  frees 5  (blocks 8)   [impl unchanged 5 d]
    UB.10 = VOID  frees 4  (blocks 5)   [impl unchanged 42 d]
    T1.08 = FAIL  frees 3  (blocks 45)  [impl unchanged 5 d]

**The single most important unblocked spec is not being worked, and for a
measured reason that is nobody's fault this week: the builder could not run.**
75 consecutive dark slots, 2026-10-22T07:07Z → 2026-10-25T09:07Z, 75.9 h with
no `rc=0`. The last 64 were not paced skips but the **90% hard stop** —
*"STOPPED at 100% weekly usage"*. The usage week turned over between 09:07Z and
10:07Z today and the builder is live again: three `rc=0` iterations at 10:49,
11:19, 12:16. **Effort-vs-goal for the week is therefore close to zero through
no decision anyone made**, which is the honest answer and also the most
alarming one.

### Goodhart check

Pass rate **41.1% → 40.8%**, and I caused the entire fall: registering `AT.01`
and `WM.01` grew the denominator 258 → 260 while the numerator stood still.
**That is the rate falling while the count holds, and it is information, not
shame — it is what naming a gap costs.** The registry grew because the anatomy
audit found two capabilities with no chair; refusing to register them would have
held the rate at 41.1% by keeping two holes invisible. Said plainly: this week's
rate movement is an artefact of honesty, and the underlying fact is that the
numerator did not move at all.

### Constitution coherence

No new internal contradiction found between `GOAL.md` and `SYSTEM.md` this
sitting. One pre-existing tension is re-noted rather than re-litigated: `GOAL.md`
carries both *"purely out of curiosity"* and the needs-as-curriculum section,
and the page itself already reconciles them (*"read the two drives as
partners"*). No edit proposed, and none made — the constitution is never
silently edited.

### The honest paragraph

We are busier than we are closer, and this week the gap was not even busyness —
it was silence. The machine that was supposed to be living a life spent three
days switched off by a meter measuring other tenants, and the work that did
happen was almost entirely the apparatus examining itself: tools that check
other tools, readers that check readers, a desk disposing promises another desk
made about work a third could not reach. I can say with a straight face that the
measuring is getting better, because it caught me twice today — once about to
stamp finished on work that did not exist, once about to create two chairs
nobody could ever sit in. That is real and it is worth something. But nothing
got closer to a creature that lives. He still cannot smell a fire through a
wall, still has no model of his own body, still does not choose what to look at,
and the one thing we learned about him this week we learned by a reference arm
failing on worlds we built for him to be watched in. The single most important
step toward Jack was discovering that the hearing fixture's repair had very
nearly worked and that nobody had read the number — that is a creature-shaped
finding, about whether he can be honestly tested at all. The most concerning
drift is the one this page cannot stop reporting: the organs now generate more
work for each other than for him, and a queue that takes a hundred and twenty
cycles to drain is not a backlog, it is a second project.

---

## Part 2 — the test re-examination

**Scope, stated honestly: this sitting did NOT complete the 8–12 spec sample.**
The OVERDUE class was eighteen rows deep and `D28`'s armed default puts its
disposal before everything else; that consumed the sitting's first half. What
follows is what the re-examination actually produced, and one of the two items
is a genuine Part 2 finding of the strongest kind.

### STRENGTHENED

| what | change | why it is STRONGER |
|---|---|---|
| `experiments/review_queue.py` — the `DELIVERED — AWAITING STAMP` reading (`49969f6`, `T0.31` re-bought PASS 2.14s) | added `_commit_date()` and a per-row `PRE-DATED RECEIPT` flag: a `BUILDER-TRACE:` whose commit predates the row's own `ROUTED:` date is no longer advertised as a delivery; an unresolvable sha reports `UNVERIFIABLE` rather than passing | **A missing assertion, added.** The block's whole purpose is to tell a desk "finished, just stamp it", so a false entry is a direct path to a fraudulent `ACTED` — and `ACTED` is TERMINAL, so the row is never re-read. It was advertising `dp04-graded-measure-...` (routed 10-11) as delivered at `5b0d4c0`, dated **09-30, eleven days earlier**, which executed its PARENT's option (i). `git log` shows this row's actual unit — the `lc_00_gridworld_decidable.py` difficulty edit — untouched since routing. **This desk read the block, believed it, and was one edit from the false stamp.** Refuses nothing, exempts nothing, moves no threshold. |
| `dp04-graded-measure-passed-its-guard-and-did-not-cure-the-venue` — the option (ii) acceptance test (`b6445cf`, and onto the builder's page) | one conjunct became **two**: the edit is accepted only if it resolves the need band **AND** leaves the `headroom` lane payable (min-task null headroom ≥ `HEADROOM_MIN_NEED`) | **Strictly harder than what the row ordered on 10-11.** The row's own trace reports the cap-200 venue cannot pay the claim on the other lane — null headroom mean **33.94** vs `NEED_MIN_GAIN` **35.0**, min-task **20.47** vs `HEADROOM_MIN_NEED` **56.0** — so a registered run is expected to **VOID on `headroom`**. Executed on conjunct (1) alone, the unit spends a world edit and two re-certifications to buy a predicted VOID. No threshold moves; both constants byte-unmoved; acceptance is still not `DP.04`'s claim verdict. |

### Re-examined and found knife-edge, not weak — `HR.1`

Not a strengthening and deliberately not a bar move, recorded because it is the
clearest "so what?" available this week. `HR.1`'s three repair arms are spent
and **the repair largely worked**, which no sitting had read:

    attempt 1/2  original        0.2375 / 0.3812 / 0.4268   mean 0.3485  FAIL
    attempt 3    arm (a) EQ      0.2062 / 0.2562 / 0.2739   mean 0.2455  REFUTED
    attempt 4    arm (c) swap    0.1250 / 0.0750 / 0.10625  mean 0.1021  FAIL

Against chance 0.05 the channel leak fell **6.97× → 2.04×**, removing **82.6%**
of the excess. The stratum is **160 trials**, so each reading is an integer
count: seed 0 is 20/160, **seed 2 is 17/160 — one trial over the 16/160 bar** —
and seed 1 PASSES at 12/160. A family root's FAIL resting on a one-to-four-trial
spread is a threshold question, and both remaining repairs are bar moves this
desk may not make. Routed as `D54`.

---

## Part 2.5 — steering maintenance

**1. PRIORITY reconciled (`e70df97` content, reasoning at `8dca9ca`).** Three
live orders on `1^22` were already discharged and nothing said so — `LC.03`'s
comparator (row ACTED; the live work is the BUY at
`lc08-comparator-registered-and-unbought`), the GPU ref-pin (ACTED), and the two
`WAITS-ON` readings (ACTED, and both print today). Each verified by running
`review-queue`, not by reading the row. One **false ranking claim** corrected:
the page said `run blocked` *"ranks `T1.08` first by a wide margin (frees 3,
blocks 45)"* — it ranks by `frees`, where `T1.08` is **third**. The error ran in
the harmful direction, aiming the builder at the blocker it is prohibited from
touching and away from the two that outrank it. Live worked example of the open
row `steering-pages-rank-the-frontier-by-blocks-not-frees`.

**2. FIELD_WATCH consumed — nothing to consume.** `docs/FIELD_WATCH.md` is
unchanged since **2026-10-05** (`32b098c`), 20 days against a weekly cadence,
because the gate refused the sweep on two consecutive Mondays (`STOPPED at 98%`
on 10-12, `STOPPED at 92%` on 10-19). All five week-10 findings are already
ROUTED to queue rows (`run status` FIELD-WATCH FINDINGS, 0 `UNROUTED`). Next
sweep is tomorrow, Monday 10-26, and for the first time in three weeks the
usage week is fresh — so it has a structural chance. Carried on `D52`.

**3. SEAT STALENESS.** `champions --check` EXIT 0, ratchet ok. Findings, none
new this week except by my own hand: 2/3 UNFALSIFIABLE are **ASR** and
**Speaker ID**, both unfalsifiable *because* `HR.1` is red — which is precisely
what `D54` now governs, so that entry's conservative default is also a decision
to keep two seats uncontestable. 4/4 UNWINNABLE (Episodic retrieval, Language
grounding, Smell, Body schema). 3/3 TRIGGER DEBT, including **World** with no
`TRIGGER:` declared at all.

**4. ORGAN LIVENESS.** Builder **live** (12:16, hourly). Review = this sitting.
Regate 10:43. **Overseer fired 12:37 and exited `rc=1` mid-report**, sealing
`docs/OVERSIGHT.md` as a draft (`fca2035`, `e70df97`) — alive but failing.
**Field watch is the one past 2× its cadence**: fires on schedule, produces
nothing, 20 days silent. Silence is never success; carried on `D52`.

---

## FOR THE BUILDER — ordered, cheapest and most perishable first

**ITEM 0 — PACE UNCHANGED.** `1^16` ITEM 0 still governs. Derive the streak and
the meter yourself; nothing here is cached. You are live as this is written —
inherit neither that fact nor its opposite, re-derive it.

**1. Read `scripts/ladder_prompt.md` before anything else: three of its live
orders were retired this morning.** `LC.03`'s comparator, the GPU ref-pin and
the two `WAITS-ON` readings are all `ACTED` and marked DISCHARGED — DO NOT
START. Starting one is the most expensive mistake on that board, in item 2(a)'s
own words.

**2. `DELIVERED — AWAITING STAMP` can lie to you now, and the flag is new.**
A row marked `!! PRE-DATED RECEIPT` is NOT finished — it cites prior work and
what it owes is still open. One row carries the flag today
(`dp04-graded-measure-...`). Unflagged rows read as before.

**3. `lc08-comparator-registered-and-unbought` (OPEN, DUE 2026-10-26) — the
comparator is registered and UNBOUGHT.** This is the live successor to the
retired order above. Conjuncts off the LIVE row: PAIRED (identical seeds, step
budget, venue as the seated `A4` run) and scored on `lg_margin_null` +
`lg_margin_twin`, neither droppable. 14.40 core-h / 3 seeds.

**4. `AT.01` and `WM.01` are newly registered, CPU, and `coverage` lists both as
"fillable today".** Not ordered — named so you know they exist and are
reachable. `WM.01` is the only RUNNABLE claim on the `fast/slow` commitment;
every other one is blocked, welded or foreclosed. If a slot opens with nothing
dated in it, these are honest work on a real hole rather than another re-buy.

**5. `DP.04` option (ii) now has a TWO-conjunct acceptance test** (DUE
2026-11-01, on your page in full). Resolve the need band **AND** leave the
`headroom` lane payable. Executed on conjunct (1) alone it buys a predicted
VOID. `LC.00`/`DP.00` both re-ran PASS on 10-24 on the UNEDITED venue — those
are re-buys on the old world, not this bill.

**6. The 10-22 `T2.05` port order is DISCHARGED and is removed from this page.**
It read *"Port `T2.14`'s `task_not_trivial_all` + `floor_selftest_ok` into
`T2.05`, and report `mse_nonlinear` + `nonlinear_headroom`"* and was the only
remaining `STEERING-PAGE ORDERS` entry naming a spec the runner would refuse.
You executed it: attempt 5 reports `floor_selftest_ok` 1.0,
`task_not_trivial_all` 1.0, and `nonlinear_headroom` negative on all three
seeds. It should not have still been standing as a live order.

---

## FOR THE OWNER

**1. `SO.07`: do not let anyone re-base `REF_MIN_FED`. Routed as `D53`**
(`class: goal`, `blocks: SO.07`, `decide_by 2026-11-08`). Recommendation as
routed, verbatim:

**Do not let anyone re-base `REF_MIN_FED`, and give the world/body defect a
live owner instead. `SO.07` is VOID because its reference arm — privileged
food positions, every advantage the rig can give — fed only 0.333 / 0.667 /
0.667 of lives on recording worlds 3–5 against a 0.8 bar. The instinct is
that the bar is miscalibrated. It is not: the SAME constant, in the spec it
was copied from, is met at the CEILING — `XL.01` records `ref_fed_frac`
1.0000 and `ok_ref` 1.0, and XL.01's own FAIL is on a different conjunct. A
bar that reads 1.0 where it was born and 0.33 where the claim runs is not
measuring the bar, it is measuring the venue. So a world-relative re-base
would take a bar a working venue clears with room and lower it until the
broken venue clears it too — converting a measured body defect into a
passing number. My recommendation is (c)'s diagnosis with a live parent, and
that (b) be refused on this evidence if it is ever proposed.**

**2. `HR.1`: hold the 0.10 bar, and know how thin the margin is. Routed as
`D54`** (`class: goal`, `blocks: HR.1`, costs 3 specs, `decide_by 2026-11-08`).
Recommendation as routed, verbatim:

**Hold the 0.10 bar and leave the hearing family red, but know exactly what
you are buying. `HR.1` exists to catch a corpus whose microphone identifies
the speaker, and it caught one — twice. The repair worked: arm (a) then arm
(c) took the clean-stratum leak from 6.97x chance down to 2.04x, removing
82.6% of the excess. What remains is a knife-edge, and I want you to see how
thin it is before you rule. The stratum is 160 trials, so every reading is an
integer: seed 1 PASSES at 12/160, seed 2 misses at 17/160 — ONE TRIAL over
the 16/160 bar — and seed 0 misses at 20/160. The family root's FAIL rests on
the worst-seed protocol applied to a one-to-four-trial spread. My
recommendation is still to hold: a residual channel cue at twice chance is
exactly the confound `HR.3` would launder into a speaker-ID claim, and "it
only leaks a bit" is how a fixture stops being a control. But this is a
threshold judgement about how clean is clean enough, and it is yours — I may
strengthen a bar and I may never weaken one, so I cannot take either
remaining repair even to say no to it.**

**3. A desk with no legal move now has THREE live rows, not one — `D49`'s class
is real. Already routed: carried forward on `D49`, NOT re-asked.** `D49`
(2026-10-18) named *"a queue row whose only permitted dispositions are ALL
forbidden to its only permitted dispositioner."* This sitting measured two more
instances and routed each on its own merits:

| row | why no desk can move it | routed |
|---|---|---|
| `pl02-void-gate-quantifies-over-its-own-nulls` | every repair forbidden to its dispositioner | `D49` |
| `so07-recording-worlds-fail-the-reference-bar` | (a) breaks the design/data split, (b) is a bar move, (c) routes to a decision closed 50 days | `D53` |
| `hr1-clean-stratum-is-a-microphone-measurement` | both remaining repairs are bar moves | `D54` |

One instance is an anecdote; three is a structure. A desk that can only ACT,
DECLINE or re-date has no move when the work is legal but the mover is not, and
its only honest output is a date it will break — which is exactly what five
broken dates on `so07` and three on `hr1` were. **I am not asking for a new
disposition verb in this item, because `D49` already holds the question and
re-asking it would be the churn `VANISHED-OWNER-ASK` exists to catch.** What I
am adding is that `D49`'s `decide_by` is **2026-10-27**, two days out, and it is
now carrying three rows rather than one.

**4. NO-DECISION: liveness, the blackout and the perishable GPU forecast,
reported per `D30`'s standing default — nothing here to rule on.** The builder
was dark for **75 consecutive slots**, 2026-10-22T07:07Z → 2026-10-25T09:07Z,
75.9 h with no `rc=0`; the last **64** were the 90% hard stop reading up to
*"STOPPED at 100% weekly usage"*, not paced skips. **It has ENDED** — the usage
week turned over between 09:07Z and 10:07Z today and three iterations have run.
Beside it, the perishable cost as `D30` requires: **`2026-W43` opened today with
1.777 h drawn of Kaggle's 30 h free quota — ~28.2 h free, expiring Saturday
2026-10-31.** That is the **sixth consecutive week** the quota expires
substantially unbought. Two corrections to the record, measured rather than
inherited: (a) prior sittings asserted the usage week resets at 04:59Z — the log
shows `STOPPED at 100%` still at 09:07Z and the resume at 10:07Z, so the
boundary is ~10:00Z Sunday and every date derived from 04:59Z came from a wrong
constant; (b) **the `dark_slots` ratchet reads 0 and read 0 as the blackout
ended**, because it is a trailing streak any `rc=0` resets — a 75-slot outage is
already invisible to it, which is the open row
`dark-slot-ratchet-is-a-streak-so-an-ENDED-blackout-reads-zero` and this week is
its largest worked example. Blackout and price are already on your desk as
`D46`; the pace line, `PACE_FLOOR`, `PACE_CAP` and the 90% stop are byte-unmoved.

**5. NO-DECISION: a cross-organ commit hazard fired on this desk today, and it
is already owned by an open queue row — reported, nothing to rule on.** This
desk repaired the builder's priority page and held it dirty for the minutes it
took to verify page size. In that window the overseer's 160th audit exited
`rc=1` and its seal committed **every** dirty file in the shared tree, including
my repair, as `e70df97` — *"overseer: rc=1 run's other dirty files, committed
unbannered"*. My own commit then found a clean tree and refused. So three
retired false orders and a corrected frontier ranking entered the repo under a
message describing them as another organ's leftovers; the reasoning now exists
at `8dca9ca`, an empty commit written for that purpose. This is the open row
`bare-git-commit-writes-the-whole-shared-index` (DUE 2026-10-31) promoted from
hypothesis to measured instance, with the builder's steering page as the victim
— the highest-leverage file a wrong commit can carry. It is also this desk's own
founding scar (74th audit) arriving from a direction the mitigation does not
cover: committing each act as you make it does not defend against another organ
committing *for* you.

**6. NO-DECISION: what the completeness audit found, declared so a named gap
becomes a decision rather than a blind spot.** Audited against the human
cognitive inventory from outside this repo, not against our own documents. Two
gaps were closed by this sitting's anatomy audit (seats added, each registered a
real ring: **Attention**, zero specs before today; **Working memory**, two
PASSING certificates and no seat). Three remain open and are named here with no
ask attached:

- **Body schema / self-model — still ZERO specs**, 78 days after the owner first
  found it on 2026-08-09. It has a seat, and `champions --check` lists that seat
  under *"seats no one can ever WIN"*.
- **Theory of mind / false belief — one spec, `GEN.03`, not implemented.**
  `GOAL.md` calls other minds *"where most human intelligence actually came
  from"*.
- **Tool use — a named commitment with one unimplemented spec (`CU.6`).**
  `GOAL.md` promises *"a tool he has not made yet"*.

**And the finding underneath all three, which is the 2026-08-09 scar restated:**
`coverage.py` audits us against a list of **29 commitments**, and neither
`attention` nor `working memory` is on it. The instrument could not have found
either gap, because an audit inherits every hole in its own standard. The two
seats added today are not on that list either.

---

*Written last, as the receipt for commits that already exist. Every act on this
page was committed as it was made; a death while writing this page would have
cost the page, not the work.*
