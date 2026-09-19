# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **DAILY** (Part 1 over the last 24 h, and Part 2.5 in full). Part 2 is
> deliberately skipped — tests are re-examined on Sundays.

**2026-09-19 06:37–07:0x UTC — DAILY, held on the first attempt.** Window: the
24 hours since 2026-09-18 18:00.

*The one sentence: **the builder wrote the dies-with-parent lesson into
`LESSONS.md` at 05:18 and broke it at 06:09 — and the organ built to notice
exactly that was wired behind a branch that never runs on a live slot — while
this desk finally did the two things it has been promising for three days:
rewrote the steering block whose opening sentence had been false for four days,
and disposed two overdue rows, the first reduction it has made to its own
violation count since the thirteen broke together on 09-13.***

> This page is the RECEIPT for commits that already exist, every one of them
> made before this page was written: `c124fad` (steering `1^9`/`2^9`),
> `5206753` (`waits-on-declared-field` DISPOSITIONED), `75acde7`
> (`hash-salt-lottery-in-a-gated-metric` DISPOSITIONED), `d2a6580` (ratchets
> record), `8b10fd7` (the trend row).

---

## Part 1 — the numbers, then the honesty

**Velocity.** 108/249 demonstrated, **43.4%** — the **fourth consecutive
zero-net day**. Ledger composition: PASS 108, FAIL 26, VOID 14, BLOCKED 1 across
149 rows carrying an attempt. Rework **78.5%** (117 of 149, unchanged). 33
commits in the window, 28 of them the builder's, 6 touching the ledger.

**The frontier.** `run next`: **44 runnable, 0 fresh** — 29 carry a settled
verdict, 15 are held by a park, a foreclosure or an open decision. Unchanged in
shape from yesterday, and the reason is unchanged: every path off that board
runs through a design answer owed by **this desk**.

**The day's finding, and it is a clean one.** The 05:18 slot committed `LT.01`'s
C2' implementation clean at `b16de57` — two-branch gameability with an
adversarial height-seeker, pilot `adv_rise` **0.7142 m** over the 0.6 m bar,
null 0.0099, 0 ladder engagements — and in the same motion wrote the **fourth**
occurrence of the dies-with-parent class into `LESSONS.md`. Fifty-one minutes
later the 06:07 slot declared `run_spec LT.01` as pid `1825829` at **06:08:31**,
ended `rc=0` at **06:09:57**, and stamped the child `EXITED` in that same
second — **86 s into a job whose attempt 1 took 1374 s.** Fifth occurrence.
`LT.01` is still `attempt: 1` at `28a232e`; attempt 2 was never bought.

**And the part that matters more than the lost run: nobody was told.**
`notice_exited_dispatches()` was shipped under the 99th audit's B6 for precisely
this event — it reads `declared_pids` for `EXITED` rows naming a
`dispatch`/`run_spec`/`detached` launch and says each loudly into `ladder.log`.
But `scripts/ladder_loop.sh:199` wires it as
`pace_gate say || { harvest_bookkeeping; notice_exited_dispatches; exit 0; }` —
**it is called only when the slot is PACE-SKIPPED.** A slot that actually runs
never calls it. So today's death, on a fully awake builder, produced no notice
anywhere, and would have been found the way the last four were: by a human
reading a log tail. That is B6's own stated failure mode, occurring inside B6.
Both are ordered to the builder as items 1 and 2 of `1^9`.

*Checked and not claimed:* `D20` fired at ~00:2x today declaring the detached
lane CLOSED to registered spec work, which reads at first glance like a
same-day violation. It is not one — `D20`'s closure is scoped to the `cpu<48h`
class and the `launch_detached.sh` lane, and `LT.01` is neither. The honest
adjacency is worth more than the false hit: the **sanctioned** detached lane has
a wall, a stamp and a ruling, and on the same morning an **ad-hoc** backgrounded
run with none of those quietly lost a unit.

**Transitive-block mass, recomputed.** `unreachable = 97` (at its declared floor
since 09-13), `goal_unrunnable = 7`, `claim_dead = 8`, `commitments_uncovered =
4` (at floor). `fail_unowned = 0` — and `D23`'s warning on that zero now reads
harder than yesterday: **24 of the 26 owned FAILs are owned by a queue row**,
i.e. by a dated promise from this desk. `gpu_hours_no_verdict` **48.42 h TOTAL**.
The ladder's block mass and this desk's backlog are one number seen twice.

**Goodhart check.** Rate flat at 43.4% on a flat registry (249) for the fourth
day. **This is neither the ladder holding ground nor outrunning itself — the
rate carried no information again today, and it is the fourth day in a row it
has not.** Four consecutive readings that cannot move is itself the signal: a
metric that is structurally unable to register certificate maintenance, design
disposal or a refused-corpus finding is not measuring this project's weeks any
more. I am not proposing to change it — a demonstrated count that cannot be
inflated is worth more than one that moves — but a chief scientist who quotes it
a fifth time without saying that would be quoting a number he knows is inert.

**Effort vs goal.** Of the window's 33 commits: ~20 served the instrument
(`T3.09`'s registry stamp, the ratchet-counter edit, `T0.21`/`T0.36` re-buys,
`D20`/`D30`'s firings, my five), ~6 served the ladder's science (`LT.01`'s C2'
implementation and pilot, the attempt-2 launch that died), ~7 journal and
lesson. The instrument-heavy ratio the desk has flagged for a week persists.

### The honest paragraph (no numbers)

The most striking thing about today is not that a run was lost — runs are lost —
but that the system wrote down the exact lesson that would have prevented it,
and then, inside the hour, with the ink wet, did the thing again. That is worth
sitting with, because it is a finding about what kind of organ `LESSONS.md` is.
It is a memory, and the system has been treating it as a control. A memory that
is read by a fresh agent at the top of every slot is a real and valuable thing,
but it competes for attention with everything else on the page, and attention is
exactly the resource a tired loop at six in the morning does not have. The
control surface is code, and the code that was supposed to catch this was built
correctly and then attached to the one branch where the event cannot happen — a
watchman posted at the door nobody uses. Against that, the honest credit: the
builder ran every slot it was given, fired two armed defaults with the required
wording and without touching a threshold, and the C2' implementation it produced
is a genuinely good piece of adversarial science that would have cleared its bar
if it had been allowed to finish. And this desk, for the first time since its
promises broke, actually reduced its own debt instead of writing about it — two
rulings that go beyond the menus they were offered, one of which found that the
option the builder called cheapest could never have fired at all. The drift to
name is unchanged and it is mine: a creature is not built out of findings about
files, and the reason the perishable hours died again last night with nothing
legal to spend them on is that the answers that would have unblocked them are
sitting on my desk with dates on them that have already passed.

---

## Part 2.5 — steering maintenance

**Organ liveness.** Ladder **06:07** — alive, and **7 of 7 hourly slots ran
today** (00:07 through 06:07, no PACING, no ABORT). Overseer **06:37** —
alive, running concurrently with this sitting as it does every morning. Field
watch **09-14 05:57** — on cadence (Monday weekly, next 09-21). Review — this
sitting, held first attempt, the repaired lane's second clean day.

**`ladder_prompt.md` PRIORITY — REWRITTEN (`c124fad`), the first act of this
sitting exactly as the 09-18 page promised.** `1^9`/`2^9` supersedes `1^8`,
whose opening sentence (*"YOU HAVE BEEN DARK FOR 18 CONSECUTIVE SLOTS"*) and
whose entire ordering (priced in W37 hours *"expiring Sat 2026-09-19"*) had been
false for four days. All four of `1^8`'s items were discharged or dead. The new
block points at living sources — it quotes `run next` as *"0 fresh of 44"* with
the standing instruction to take counts from the tool, never from the page. Its
three items: run `LT.01` attempt 2 **in the foreground**; fix the
`notice_exited_dispatches()` wiring and re-buy `T0.33`; refuse `W38`'s opening
inventory in advance, because the absence of a legal buyer is my debt.

**`FIELD_WATCH.md`** unchanged since 09-14 (`9075d58`) — nothing to consume.
Week 7's two findings remain ROUTED and cited, 0 `UNROUTED-FIELD-FINDING`.

**Seat staleness.** `champions_trigger_debt = 3` (unchanged since 09-03),
`champions_unwinnable = 4` (at floor since 09-13). Neither moved; both remain
this desk's standing debt and neither is due today.

**Queue — the first real movement in the desk's favour.** Started the sitting at
**14 OVERDUE**, ends at **12**. `review_queue_violations` 10 → 12 recorded in
`d2a6580`, with both movers named: **+4** the midnight rollover of four more
DUE-09-18 rows (a clock movement, the fourth consecutive night this desk has
paid it) and **−2** the two disposals below. `review_queue_net_arrivals` 8
(clock +0, act −1) — **not mine**, it already read 8 before my first commit, and
it is recorded rather than claimed.

---

## DISPOSED THIS SITTING

**1. `waits-on-declared-field` → DISPOSITIONED (`5206753`), re-dated 09-21.**
The grammar question is answered **yes**: a live non-`HELD` row may declare
`WAITS-ON: <row id>`. The proposal's core judgment is right — coupling and
ageing-exemption are different things and `BLOCKED-BY:` conflates them. These
rows *should* age; `WAITS-ON:` buys nothing, and that emptiness is the feature.
The **base** proposal is refused on the builder's own case against it: an
optional unenforced field is written by whoever remembers, and a grouped count
from partial declarations is *confidently wrong*, worse than today's absent
line. **The cheaper variant is adopted with one repair, because as written it
could never have fired**: *"print only when EVERY row in a pile declares one"*
is unsatisfiable for a genuinely independent row, which has no root to name — so
the line never prints and the feature is inert. `WAITS-ON: none` becomes an
explicit declaration of independence, and the grouped line gates on every live
row carrying an **explicit** declaration. That converts *"did the router
remember?"* into a completeness question the instrument answers mechanically —
the same move `decisions.py` makes when it refuses to guess which owner items
are asks and demands a written `NO-DECISION:`. **Silence is reported; exemption
is written down.** Strengthen-only: declaration-only, may touch no number
already printed, a `WAITS-ON:` naming a nonexistent row is a new VIOLATION, and
an unmet gate must print *why* it is unmet. Stales and re-buys `T0.31` as a
strengthening (18 → 19 properties).

**2. `hash-salt-lottery-in-a-gated-metric` → DISPOSITIONED (`75acde7`),
re-dated 09-21. None of the three options offered.** `(iii) NOTHING` is refused
on the row's own evidence: the verdicts survived only because both specs miss
`match` and `unanimity` by a mile, so `swap_agree` was never binding — luck
about which conjunct was slack, and the same lottery was one tie from deciding a
**seat**. `(i) THE STATIC AST SCREEN` is refused too, and **the builder's
restraint around `D27` was right**: shipping a second unmeasured heuristic while
`D27` asks whether screens work at all — carrying 104-of-107 flagged, 3-of-12
hand-checks real — would spend the exact credibility `D27` is pricing. **But
the menu carries a false constraint.** `(ii)` was priced out for one reason,
*"it doubles every spec's cost"*, and **it does not need to run on every spec**.
So: **(iv) run the exact second-salt differential, but only where the metric
DECIDES** — BINDING (within a declared margin of its gate) or ELIGIBILITY (a leg
in a seat race or `bakeoff.py` tie-break, unconditionally, since a tie-break is
decided at zero margin by construction). **(iv) is stronger than (i) and does
not collide with `D27` at all**: a static screen is a heuristic with a
false-positive rate to calibrate; a differential re-run is a **measurement** —
exact, zero false positives, nothing to write down — so it may ship whichever
way `D27` falls. Ordered: **measure the deciding set and its cost against
`CPU_DAY_CEILING_S` and report before implementing**; if it does not fit, do not
trim the rule to fit — the ELIGIBILITY half ships alone. Margin declared in
source before scanning, never tuned after seeing what it captures. **No bar
moves in either direction**: a salt-dependent binding metric is repaired to
determinism, never covered by its gate. CPU classes only.

---

## FOR THE BUILDER

Ordered, and the order is in `scripts/ladder_prompt.md` `1^9` as the binding
copy — this is the summary, that is the instruction.

1. **`LT.01` attempt 2, IN THE FOREGROUND.** Check `LT.01`'s `attempt` first; if
   it is still 1 at `28a232e`, it is still owed. ~23 min of CPU fits a slot with
   room. On landing: read the branch, commit the row by name, stamp
   `lt01-c2-body-cannot-rise` ACTED with the executing commit, journal. **The
   0.6 m bar does not move in either branch.**
2. **Wire `notice_exited_dispatches()` onto the live path**, not only behind
   `pace_gate`. One line of wiring, noticing only, `add -A` ban intact; `T0.33`
   certifies `ladder_loop.sh`, so re-buy it in the same motion.
3. **Refuse `W38`'s opening inventory.** No legal buyer exists and manufacturing
   one is not the repair. If it is still unbought on Sunday, that is a finding
   about **me** and I will write it as one.
4. **Implement the two dispositions above when their dates come (09-21)** —
   `WAITS-ON:` with `none` permitted, and the `(iv)` measurement **before** the
   `(iv)` implementation. Do not start either early and do not fold them
   together.
5. **Still do not pre-empt** `A4`, `T2.10`'s repair, `SO.07`, `SO.10`, `T1.08`'s
   pipeline repair, `HR.1`'s fixture redesign, or `UB.10`'s successor arm. All
   mine, all overdue. Naming candidate arms, as you did on `HR.1`, remains
   welcome and is not the same as choosing one.

---

## FOR THE OWNER

**1. NO-DECISION: liveness and the standing `D30` report, nothing here to rule
on.** All four organs are alive and on cadence: ladder **7 of 7 slots today**,
overseer 06:37, field watch 09-14 (next 09-21), Review this sitting. **Dark-slot
streak: 0** — `D30`'s standing report, counted from `/data/jack-logs/ladder.log`
and not from anyone's summary; nothing exceeds 2× the hourly cadence and there
is no blackout to declare. **Beside it, as `D30`'s default requires, the
perishable price:** `2026-W37` closed overnight with **~27.78 free Kaggle hours
expired unspent** and **no legal buyer**, which the 02:0x audit confirmed
correct. **`2026-W38` opens tomorrow, 2026-09-20, with 30 free hours that expire
Saturday 2026-09-26.** There is still no legal buyer, because every GPU-class
pilot is blocked behind a Review design row. That is a fact about item 2, not a
new ask.

**2. `D28` — cited, not re-asked (`decide_by` 2026-09-21), with the piece of
evidence it did not have.** `D28` measures this desk's drain as UNBOUNDED and
asks you to price the repair. What today adds is **what the drain actually
costs in a currency that perishes**: the backlog is not merely embarrassing
paperwork, it is the reason ~27.78 GPU-hours died last night with nothing legal
to spend them on, and the reason the same will happen to 30 more next Saturday
unless it changes. The queue and the idle quota are the same problem billed to
two different accounts. I note also, in fairness to the entry rather than to
myself, the first counter-evidence in its favour: this sitting disposed **two**
rows against a demonstrated rate of ~1, so the capacity `D28` doubts is not
zero — it is just far below the arrival rate. Recommendation on the entry
unchanged. I also note the overseer's armed reclassification notice (fires
2026-09-21 absent your word), and I do not object to it.

**3. `D27` — cited, not re-asked (`decide_by` 2026-09-20, tomorrow), and today
this desk voted on it with its hands.** `D27` asks whether this repo should buy
mechanical screens, against a measured 104-of-107 false-positive showing. Faced
with a live instance of exactly that trade in
`hash-salt-lottery-in-a-gated-metric`, I **declined the screen and chose a
measurement instead** — a differential re-run has no false-positive rate because
it does not guess. That is one data point, not an argument for your answer, and
it cuts in a specific direction worth naming: the reason `(iv)` was available at
all is that the question was narrow enough to measure exhaustively. Where a
question is not narrow enough, `D27`'s trade is still live and still yours.

**4. NO-DECISION: a defect class at five occurrences, reported because it is
about how this system learns, not because there is a fork in it.** The
dies-with-parent class has now recurred five times, and the response each time
has been to write a better `LESSONS.md` entry. Today gives the cleanest possible
measurement of what that response is worth: the fourth entry was written at
05:18 and the fifth occurrence happened at 06:09, by the same loop, 51 minutes
later. **`LESSONS.md` is a memory, and it has been doing duty as a control.** I
am not asking you to rule on this — the specific fix is cheap and is already
ordered to the builder as item 2, and one clean instance does not justify a
structural change to how this project records what it learns. I am putting it on
your desk because if a sixth occurrence follows a fifth lesson, the conclusion
will not be about `ladder_loop.sh` any more, and I would rather you had seen the
fifth than met it at the sixth.
