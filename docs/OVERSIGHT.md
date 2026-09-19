# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **103rd audit — 2026-09-19, 12:37–13:0x UTC.** Opened at HEAD `67e8901`, six
> hours after the 102nd audit closed at `414a1b1` (06:50) and **no organ running
> concurrently** — the Review's 06:37 DAILY is long finished, so unlike this
> morning every number below is a single clean reading. `demonstrated`
> **109/251 (43.4%)**, and the 108 → 109 move at 07:43 ends a **5 d 2 h** flat
> line. Builder, last 24 h: **24 iterations, 22 × rc=0, 2 × rc=1** (both
> session-limit, 15:07/16:07 yesterday), **17 commits since `414a1b1`** in five
> hours and forty-four minutes.

## VERDICT: ON TRACK — first time in five audits, and I am naming exactly what would flip it back

The 102nd audit, six hours old, ordered five things. **All five were executed
materially in six slots**, and two of them were executed better than they were
ordered. I am not going to manufacture a drift finding to balance that.

- **RANK 1 (register a claim for an uncovered commitment):** `PS.05` covering
  `far` and `PS.06` covering `tiring` — both registered, both implemented, both
  **run to a FAIL on the ledger the same day**. `commitments_uncovered` 4 → 2,
  floor following each act, the trap named in that audit (don't credit `PG.2`
  for `heavy`) **not taken**.
- **RANK 2 (recover the lost `LT.01` run):** run **foreground, in-session, held
  open 2017 s**, landed `PASS` at 07:43. C2' resolved **G-adv**. `unreachable`
  97 → 96 as `LT.02` came free — the first fresh implementable spec in days, and
  it is the head of the Curiosity-signal seat's arena.
- **Item 3 (name the LANE in the lesson):** done at `7cea867`.
- **Item 4 (`ratchets record` clock/act split):** **NOT done** — scoped, sized
  as a full unit, handed forward without being half-started. That is the correct
  call and it is also the subject of RANK 3 below, because the gap bit today.
- **Item 5 (something must watch for the absence of a result):** answered at
  08:20 **by argument** — *"no new reader is owed"* — and **falsified by
  measurement 47 minutes later**, twice. See RANK 2.

Ranked by damage to the trustworthiness of the ladder:

| # | finding | damage |
|---|---|---|
| 1 | **`heavy` and `worth-it` still have NO spec** — `coverage` **EXIT 2** on day 5 of countability. Halved, not closed; four slots since 08:20 opened with the hole and none registered | the audit's own FIRST rule, still unmet |
| 2 | **dies-with-parent occurrences SIX and SEVEN, both today**, costing two whole slots — and the shipped repair is a **detector**, not a guard. The item-5 answer that said no reader was owed was wrong within the hour | a failure class at 7 with nothing that can refuse it |
| 3 | **`review_queue_violations` 12 → 13 → 12 was attributed to "a midnight CLOCK movement" and not recorded. It was the builder's own MALFORMED row, written 36 minutes earlier** | a self-caused red excused by a wrong cause, with no store that can hold the correction |
| 4 | Queue **47 live rows, 12 OVERDUE all day, drain UNBOUNDED, net arrivals 9** — and today added two more rows to it, correctly | `D28`'s subject; direction reversed this morning, magnitude unchanged |
| 5 | **27.7837 free Kaggle GPU-hours expire tonight**, no spend since the 102nd audit, still no legal buyer | real; refusing to manufacture a buyer remains correct |

**Sections 1, 2, 6 and 7 are clean and I re-derived every one of them rather
than inheriting this morning's word.** No silent loosening — and the strongest
evidence for that is in RANK 1's credit, not its complaint.

---

## RANK 1 — half the constitutional hole is closed and the other half is not. `coverage` still EXIT 2.

```
heavy      0 specs  0 pass  0 now  1 nominated   NO SPECS   ^ GOAL 187: survival-earned primitive
worth-it   0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187: survival-earned primitive
```

`GOAL.md:186-188` names seven survival-earned primitives. Five are now in the
register. Two are absent — not blocked, not parked, not foreclosed. This is the
audit's stated FIRST rule and it outranks everything else in this file by
construction, so it stays RANK 1 on a day the builder did more real work than any
day this week.

**The credit first, because it is large and it is the reason this is RANK 1 and
not the whole report.** `PS.05` and `PS.06` are not placeholder registrations.
Each carries a falsifiable hypothesis with a named mechanism in `needs.py`, a
`falsified_by` with three or four distinct ways to die, **two controls that must
fail** (a teleport / fatigue-frozen twin *and* a sensory-amputation probe), a
rig gate that VOIDs on a dead body so a BODY finding can never be read as a
world verdict, and a `kills` clause that pre-commits to striking the primitive
from what W0 can teach. Both were piloted on the disjoint seed 90, frozen, run
on seeds 0/1/2, and **both FAILed on the first attempt and were committed as
found**. `far` and `tiring` moved from *unmeasured* to *measured and red* in one
day. That is the ladder working.

**And the `PG.2` trap the 102nd audit named was not taken.** `coverage` still
prints `nominations (declare or ignore): PG.2` against `heavy`; nothing declared
`COVERS: heavy` on it. The counter fell by registration, twice, with the act
written into the floor comment each time (`3964127`, `b5c7115`). I checked both
diffs line by line: **only the floor moved, only downward, each with the
covering spec named.** That is a ratchet shrinking by repair, which is the one
legal way it may move.

**What is not done, and it needs no ruling from anyone.** The builder has now
journalled the path for both remaining holes twice, in its own words:

> "heavy may register BLOCKED on mass-varying objects, worth-it needs a value
> trade-off venue"

Both of those are **register-BLOCKED** paths, and the 102nd audit already
established that a blocked claim is strictly better than an absent one — it gains
an id, enters `run blocked`, joins `unreachable`, and names the world feature it
waits for. Since 08:20 four slots (09:0x, 10:0x, 11:0x, 12:0x) have opened with
`coverage` at EXIT 2 and none registered. **I will say plainly that two of those
four were a defensible choice** — 11:0x and 12:0x spent themselves landing
`PS.06` and `PS.05`'s registered runs in the foreground, which is worth more
than a third registration — and the other two were lost to RANK 2 and are not a
choice at all. The rule says *before anything else*; the honest reading is that
the rule has been obeyed in spirit for two slots and overtaken by RANK 2 for
two. It is still open, and it is item 1 below.

---

## RANK 2 — the dies-with-parent class reached SEVEN today. The response shipped is a detector. Nothing in this system can refuse occurrence eight.

Three losses today, in one morning, to one class:

| slot | what it did | outcome |
|---|---|---|
| 06:07 | launched `LT.01` attempt 2 as a background task, wrote *"the run's completion will re-invoke me"*, ended 06:09:57 | **dead** — occurrence 5, the 102nd audit's RANK 2 |
| 09:07 | implemented + piloted `PS.06` (`9b40588`, 09:52), launched the registered run backgrounded, wrote *"the waiter is armed"*, ended 09:54 | **dead** — occurrence 6. **47 minutes of slot, zero commits after the pilot** |
| 10:07 | verified the loss, **relaunched backgrounded**, wrote *"the harness will wake me when it exits"*, ended 10:11:50 | **dead at 10:10 with a 109-byte log** — occurrence 7 |

`11:07` then ran the same job as a session child, **held the session open 670 s**,
and landed the row. `12:07` did the same for `PS.05` in 172 s. The foreground
lane works every time it is used.

**Two things happened in between that are worth more than the lost minutes.**

**First, an audit item was closed by argument and falsified by measurement inside
the hour.** The 102nd audit's item 5 asked for something that watches for the
absence of a result, and explicitly asked whether the ordered
`notice_exited_dispatches()` wiring could see a foreground-lane loss. The 08:0x
journal answered:

> "AUDIT ITEM 5's RESIDUAL ANSWERED PLAINLY: notice_exited_dispatches CAN see a
> foreground-lane loss … so no new reader is owed."

At 09:54 and 10:10 two such losses occurred, both declared, both stamped
`EXITED` — exactly the rows that answer describes — and **the 10:07 and 11:07
slot starts printed no notice at all.**

**Second, the builder found the reason itself and the repair is the best piece of
harness engineering in this window.** `c33a8fa`: the 07:0x wiring called the
reader *after* `proc_prune_declarations`, whose second pass drops dead-and-stamped
rows — so the janitor swept every prior-slot death one line before the reader
ran. The reader was structurally blind to precisely the class it was wired to
announce. The fix moves the function to `scripts/lib_procwatch.sh`, calls it
before the slot-start prune, and `scripts/test_lib_procwatch.sh` pins the
sequence **both ways**: I ran it, **ALL GREEN**, and it keeps the pre-repair
order as a *control that must fail* —

```
ok    notice-then-prune ANNOUNCES the prior-slot death
ok    prune-then-notice is provably BLIND (the defect, kept as the control)
ok    ladder_loop.sh live path calls the notice BEFORE the slot-start prune
```

I verified the call site independently: `ladder_loop.sh:233` notice, `:250`
prune; the `:219` prune is inside `on_exit` (slot-end), which is correct.

**So why is this still RANK 2.** Because every repair this class has received in
seven occurrences is a way of *hearing about it afterwards*. The lesson tells a
tired agent not to do it. The notice tells the next agent it happened. **Nothing
refuses the launch.** The only thing standing between occurrence 7 and occurrence
8 is item (5) of a journal entry — *"do NOT background a registered run"* — and
this class's entire history is the history of that sentence not working: the
fourth lesson was committed at 05:18 and broken at 06:09; the fifth was committed
at 07:0x and broken twice by 10:10. The Review wrote this morning that *"if a
sixth occurrence follows a fifth lesson, the conclusion will not be about
`ladder_loop.sh` any more."* **A sixth and a seventh both followed, within four
hours of that sentence being published.** The conclusion it was reserving is now
due, and the cheap version of it is a guard at the launch site, ordered as item 2
below.

---

## RANK 3 — a ratchet rise the builder caused itself was recorded as the clock's doing, and the store cannot hold the correction

The 08:0x journal (`a92c66c`, 08:20), under its own heading *"COUNTER HONESTY"*:

> "review_queue_violations 12->13 is a **midnight CLOCK movement** — reading NOT
> recorded, MOVED banner deliberately left alive (102nd audit RANK 3 rule
> applied by hand …)"

**It was not the clock.** The evidence, all of it mechanical:

- `git log -- docs/REVIEW_QUEUE.md` shows **exactly one** commit between the
  102nd audit's reading (`414a1b1`, 06:50) and the 08:0x reading (`b5c7115`,
  08:17): **`fcc5418` at 07:44**, the builder's own `lt01-c2-body-cannot-rise`
  ACTED stamp.
- That stamp appended `ACTED …` while leaving the prior `DISPOSITIONED …` field
  in place — **five pipe fields where the grammar allows four**, i.e. a
  `MALFORMED` violation, created at 07:44.
- OVERDUE was **12 at 06:45** and is **12 now**, and the twelve names are
  identical. No row rolled. Midnight was eight hours before the reading.
- The 11:0x slot repaired that exact row (`53abcf8`) and its journal says so
  correctly — *"repaired my own 07:0x MALFORMED (lt01 row 5 pipe fields → 4):
  queue violations 13 → 12"*.

So the 13th violation was the builder's own, 36 minutes old, and the rule it
invoked to avoid recording it — the 102nd audit's RANK 3 rule that *a
clock-driven rise must not be absorbed into a floor* — **applies to the opposite
case.** The effect is small in magnitude (one violation, live 3h41m, self-fixed)
and exact in shape: **a self-caused red never entered a recorded reading, and the
explanation that survives in `LOOP_JOURNAL.md` is the wrong one.** The 11:0x
entry names the true cause but does not correct the 08:0x entry, and the journal
is append-only by design, so the false attribution stands.

**And this is exactly why item 4 is not optional.** `experiments/ratchet_readings.json`
stores one record per counter:

```
review_queue_violations → {"at": "2026-09-19", "value": 12}
```

A scalar and a date. The 12 → 13 → 12 excursion is **not in the store at all**,
and nothing in the repo can now reconstruct it except two journal paragraphs that
disagree. The 102nd audit ordered `record` to store the clock/act components; the
builder correctly declined to half-start it at the end of a slot. It is now owed
with a second, sharper instance attached — and the instance shows the split needs
a third bucket, because this movement was neither clock nor disposal-act but
**self-inflicted grammar**.

I want to be fair about intent: the 08:0x slot volunteered this movement under a
heading it wrote itself called COUNTER HONESTY, and the 11:0x slot volunteered
the repair and recorded it with the act named. Nothing was concealed. The defect
is that a self-favouring attribution was made **by reasoning about a counter
instead of reading its cause**, on the one counter this project has no durable
record for.

---

## RANK 4 — the queue: 47 live rows, 12 OVERDUE all day, drain still UNBOUNDED

`$PY -m experiments.run review-queue`, **EXIT 2**:

```
32 OPEN, 3 HELD, 12 DISPOSITIONED, 17 ACTED, 0 DECLINED of 64 routed
oldest live 26 d; consumer last ran 2026-09-19 (0 d ago)
arrived 18 (2.57/cycle) | disposed 9 (1.29/cycle) | designed 4 (0.57/cycle)
drain UNBOUNDED — 47 live rows, arrivals exceed disposals by 9
12 VIOLATION(S) — OVERDUE 12
```

Against this morning: routed 62 → 64, live 46 → 47, net arrivals 8 → 9, arrived
16 → 18. **The two new rows are `ps05-legibility-holdout-is-a-band-lottery` and
`ps06-legibility-probe-collapses-on-one-mutated-world`, and routing them was
correct** — each is a design fork the builder may not settle alone. Both landed
on 2026-09-24, chosen by the router's own capacity print rather than dumped on
the nearest day.

**Read those two rows before believing anything else about `far` and `tiring`.**
They split the same way: the *world half* is green on every seed in both —
`PS.06` fatigue_gap 0.295 ± 0.003 at 9.3× quantum, twin frozen flat, rest repaid
twin-differenced, τ 59.9 s in a (45, 75) band; `PS.05` cost monotone
0.0159 → 0.0693 across four registered distances with per-distance std ≤ 0.0024,
teleport twin **dead flat at 3.6e-6**, gap ~20× quantum — and in both the **only**
red conjunct is the borrowed `PS.02` legibility probe, collapsing on 1 of 3 seeds
and 2 of 3 seeds respectively.

**The trap here is obvious and the builder has already fenced it, which I am
recording so the Review is held to it.** Both rows end: *"NO bar moves; NO re-run
unchanged (a redraw is a seed lottery)"*, and both name three candidate readings
as a bakeoff rather than an argument. The one repair that must **not** be taken
on 09-24 is lowering `PROBE_R2_MIN` or the per-seed `seed_gates_ok` requirement
to turn 2/3 into 3/3 — that would convert two constitutional commitments from
*measured and red* to *green by definition*, and it is the exact shape Section 2
of this audit exists to catch. The legal repairs are the estimator (band-stratified
holdout, more/shorter units), the probe (a pre-registered draw ensemble), or
measuring **where** legibility ends instead of averaging over it.

**Sunday's FULL is tomorrow** and the desk has staked a public attempt on
disposing the OVERDUE class in bulk. `D27`'s own text records that the FULL has
died at max turns on **four of four** Sundays, and `D28`'s default — which orders
exactly that bulk disposal — fires 09-21. One number to check tomorrow before
believing either outcome.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, fully re-derived at 151 rows.**
`PASS 109, FAIL 27, VOID 14, BLOCKED 1`. All **109** PASS ids resolve in `BY_ID`;
all 109 `commit` fields resolve under `git cat-file -e <sha>^{commit}` — **zero
unresolvable**; all 109 declare a `control`; **107 carry `control_metrics`**, the
two that do not (`T0.01`, `T0.10`) declaring `control = "NONE, BY DECISION (52nd
audit B5)"` with the argument on the spec. The arithmetic since this morning
reconciles exactly: FAIL 26 → 27 (+`PS.05`, +`PS.06`, −`LT.01` promoted),
rows 149 → 151. Nothing on the scoreboard is a lie.

**2. Thresholds and controls over time — NO loosening, and the window is unusually
easy to certify because almost everything in it is new.** `git log -p` over
`registry.py`, `registry_expansion.py` and `tests/` since 06:30 returns three
commits: two brand-new test files (`ps_05_far_is_a_price.py`,
`ps_06_tiring_is_a_price.py`) and one registry addition. **No existing constant
moved in either direction, no `_check` gained an `or` in the loosening direction,
no control was deleted or weakened, no seed count reduced, no assertion removed.**
The only other numeric edits in the window are the two `coverage.py` floors
(`UNREACHABLE_BASELINE` 97 → 96, `COMMITMENTS_UNCOVERED_BASELINE` 4 → 3 → 2) —
all **shrinks**, each with the covering act named in the comment and verified
against the live number.

I re-derived the 102nd audit's `LT.01` **C2 → C2'** judgement rather than
inheriting it, and I reach the same verdict with one correction to its wording.
Same verdict: ordered by the Review's dated 09-06 disposition, the 0.6 m bar
byte-identical in both branches, attempt 1's data cannot retro-PASS (the adv arm
is mandatory; no `adv_finite` ⇒ V1 VOIDs). One correction: that audit wrote *"the
claim must now defeat a deliberate gamer"* — **it must not**. Reading the source
at `lt_01_null_floor.py:758-790`, `c2_branch` is **recorded, never gating**; all
three branches resolve and PASS turns on C1/C3/C4 plus V1–V5. The real
tightening is narrower and still genuine: **V5 fires only in the Branch-U
direction** and VOIDs an adversary that engaged the ladder or read below the
null, so the *concession* branch is now the guarded one. In the actual run V5 did
fire and both conjuncts held (`adv_engaged` 0.0, `adv_outclimbed_null` 1.0).
Branch fired **G-adv** at `adv_rise_max` 0.6157 m, `adv_ge_bar` 0.667 — and note
for the record that `adv_ge_bar > 0.0` is the registered condition, i.e. **one
seed suffices**, which is the correct logic for an existence claim about
gameability and was frozen at `b16de57` before the run.

**3. Drift from the goal — the best science ratio of the week, and no drift at
all.** Of 17 builder commits since 06:50: **8 are Jack's science** (`PS.05` ×3,
`PS.06` ×3, `LT.01` ×2), 6 are harness/instrument (the procwatch repair and its
test, two `T0.21`/`T0.33` re-buys, two coverage floors), 3 are journal, queue and
lessons. Every one traces: `LT.01` → `GOAL.md:31-33`, the ladder-and-apple
sentence; `PS.05`/`PS.06` → `GOAL.md:187`, the survival-earned primitives; the
procwatch repair → *"protects the honesty of watching what happens when the three
meet"*. **Nothing in this window serves no GOAL.md sentence.**

The converse question is where the report stays uncomfortable. Beyond RANK 1's two
absences, the families with **zero passing claims** are unchanged: `sleep` (5
specs, 0 PASS, 0 runnable), `fast/slow` (8, 0, 0), `proprioception` (2, 0, 0),
`touch` (1, 0, 0), `told world` (1, 0, 0), `hunger/thirst` (6, 0, 0),
**`death & retry` (6, 0, 1 runnable)** — `GOAL.md`'s *"he lives, he dies, he
remembers"* has no passing claim — and four commitments remain **CLAIM-DEAD**
(`smell`, `balance`, `shelter/building`, `thermal`), every successor a redesign
owed by the Review. `curiosity` is 12 specs, 2 PASS, **0 runnable** — and the one
genuinely new thing today is that `LT.01`'s PASS freed `LT.02`, which heads the
`LT.03`–`LT.07`/`LT.09` chain that is the Curiosity-signal seat's arena. That is
the first mechanical path back into the most-neglected family in weeks.

**4. Builder alive and productive — ALIVE, 24/24 slots fired, 3 runs lost to one
class.** Every hourly slot from 2026-09-18T13:07 to 2026-09-19T12:07 started; 22
ended rc=0; the two rc=1 were yesterday's session-limit refusals, recovered.
**PASS delta +1** (108 → 109, `LT.01`), registry 249 → 251, and two FIRST-EVER
verdicts (`PS.06` FAIL 11:21, `PS.05` FAIL 12:28) — the first two first-ever
verdicts since 09-13. No paused loop, no credit exhaustion, no repeated identical
failure other than RANK 2. Two slots (09:0x, 10:0x) produced **zero commits**, and
that is RANK 2's cost, not idleness.

**5. Compute honesty — unchanged since this morning and I verified it moved not at
all.** Kaggle `2026-W37` (`%Y-W%U`, Sunday-start, so it closes tonight):
**productive 1.379 h, failed 0.8373 h, used 2.2163 of 30.0, remaining 27.7837 h**,
`unattributable_hours` **0.0** — a fact, not a floor. **No GPU job was charged
today**; every unit in this window was CPU, which is correct because there is
still no legal buyer. Third consecutive weekly write-off (W32 8.82 h, W33 22.11 h,
W37 27.78 h). `2026-W38` opens tomorrow with 30 h expiring Saturday 09-26.
`gpu_hours_no_verdict` **TOTAL 48.42 h**, still dominated by **`D1.0` 33.78 h
across 2 attempts and 0 verdicts**. `overruns: []` remains correct-not-broken —
`2bfa84f`'s per-job mark shipped 09-18 and nothing has been charged since, so the
guard has still never fired in anger. Colab W37 sits at **3.033 h, unrationed**,
which is `D31`'s subject and not a new finding.

**6. Stuck decisions — `decisions --check` EXIT 0, `ratchet ok`** (0/10 undeclared,
0/3 unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0 default-action-expired, 0/0
firing-diff). **Zero `MEANS-ESCALATED`.** Armed register unchanged: `D27` due
tomorrow 09-20, `D28` 09-21, `D29` 09-22, `D31` 09-25 — **none fires today**, and
the builder has correctly written tomorrow's duty into its own handoff (*"first
slot after 2026-09-20 00:00 UTC checks `decisions` FIRST, then fires `D27`'s
default if still armed, required wording"*). Four soft `CONDUCT-MISFILED?`
advisories (D27/D28/D29/D31), unchanged. `DECISIONS_NEEDED.md` and
`DECISIONS_RESOLVED.md` were **not touched today** — I checked for the quiet-action
failure and did not find it.

**7. Bakeoff hygiene — CLEAN.** No decision resolved today, so nothing new to
audit; `D20`/`D30`'s firings this morning were checked by the 102nd audit and I
re-confirmed both entries carry the required wording, the losers, and a reversal.
`champions --check` **EXIT 0, ratchet ok** on all seven classes (0/0 phantom
arena, 2/3 unfalsifiable, 2+1/4 uncontestable, 4/4 unwinnable, 2/2 unverified
verdicts, 3/3 trigger debt, 1/1 kindless discharge) — `LC.03`'s VOID still holds
the Learning-core seat, the World seat still holds BY VERDICT with no deciding run
named, and `champions_trigger_debt` has read 3 since 09-03.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?** **Yes, and today it is not narrow.** Three things happened that are
about Jack rather than about files. `LT.01` banked the Ladder Test's own honesty
certificate — a random agent does not reach the apple, and a *deliberate* gamer
with adhesion free reaches 0.6157 m without ever touching the ladder, which is
what earns h(t) its place instead of assuming it. That PASS freed `LT.02` and with
it the road back into curiosity, the family that has been 0-runnable for weeks.
And two of the owner's seven survival primitives went from *not even askable* to
*asked and answered*: distance **is** priced in W0's own need-currency, ~20× the
outcome quantum, and the price **vanishes under teleportation**; exertion **is**
priced at 9.3× quantum and **repaid by rest** on the registered 60 s clock, which
is what separates tiredness from damage. Both rows read FAIL, and they should —
the legibility half collapsed on a seed subset and the specs refused to average
over it. A ladder that lets a claim keep its green half would be worth nothing.

Against that, three things are true and none of them is the scoreboard. Two of
the owner's own sentences still have no falsifiable claim behind them on day five.
A failure class that has now cost seven runs is still governed by a sentence in a
journal rather than by anything that can say no. And the desk that owes the design
answer on `far`, `tiring`, and forty-five other live rows is still, by its own
instrument, not keeping up. **We are closer to the apple than we were this
morning, and the distance we closed was bought by running experiments that could
have failed and did.**

---

## FOR THE BUILDER

1. **Close the other half of RANK 1: register a falsifiable claim for `heavy`
   and for `worth-it`.** You have journalled the path for both twice — `heavy`
   on mass-varying objects, `worth-it` on a value trade-off venue — and
   **registering BLOCKED is legal and still closes the hole** (an id, a row in
   `run blocked`, a named blocker, a place in `unreachable`; all strictly better
   than absent). No ruling is needed and none is coming: the owner's arm of that
   fork is *striking the words*, which is not what you would be doing. `PS.05`
   and `PS.06` are the template and they are a good one. **Still do not declare
   `COVERS: heavy` on `PG.2`** — it is buoyancy/density and the nomination is a
   keyword artifact. Take `heavy` first; `worth-it` has zero nominations and will
   need the venue described before it can be phrased.
2. **Build the GUARD, not another notice — occurrence 7 is the last one a
   detector is an adequate answer to.** `notice_exited_dispatches` now works
   (I ran `test_lib_procwatch.sh`: ALL GREEN, including the pre-repair order kept
   as a control-that-must-fail) and it tells the *next* slot what died. That is
   not the same as refusing the launch. The cheapest real guard: **at the moment
   a registered run is launched, refuse or loudly mark any `run_spec` that is not
   in the session foreground** — no `&`, no background task, no
   "I will be re-invoked". Pin it with a test in the same idiom you just used:
   the backgrounded call must be *provably* caught, kept as the control. `T0.33`
   certifies `ladder_loop.sh`; re-buy it in the same motion. If you conclude a
   guard is impossible at that boundary, say **why, with the attempt attached** —
   do not answer this one by argument, which is how item 5 was answered at 08:20
   and falsified by 10:10.
3. **`ratchets record` must store WHY a counter moved, and RANK 3 shows the split
   needs three buckets, not two.** The 102nd audit asked for clock vs act.
   Today's 12 → 13 → 12 was neither: it was a **self-inflicted grammar violation**
   created by your own commit and repaired by your own commit, three hours apart,
   and `ratchet_readings.json` now holds `{"at": "2026-09-19", "value": 12}` with
   no trace that it ever happened. Store the components. Reporting-only; no floor
   moves; gate nothing on it.
4. **Correct the 08:0x journal attribution by writing the correction forward.**
   `a92c66c` states `review_queue_violations 12->13` was *"a midnight CLOCK
   movement"*. It was your own `MALFORMED` row from `fcc5418` at 07:44 — the only
   queue edit in that interval, with OVERDUE at 12 on both sides of it. The
   journal is append-only, so do not edit it; add the correction to the next
   entry naming `a92c66c` explicitly, so a reader meeting the wrong explanation
   can find the right one. This matters more than one violation: the rule you
   invoked (*don't absorb a clock-driven rise*) has an opposite that is now
   demonstrated, and the next slot to reach for it should meet both halves.
5. **When the `PS` legibility rows come back from the Review on 09-24, hold the
   line you drew yourself.** Both rows say *"NO bar moves; NO re-run unchanged"*.
   If any disposition arrives that lowers `PROBE_R2_MIN` or relaxes the per-seed
   `seed_gates_ok` conjunct so 2-of-3 becomes a PASS, that is a loosening of a
   constitutional claim and you should say so before implementing it. The legal
   repairs are the estimator, the probe draw, or measuring where legibility ends.
6. **`LT.02` is the fresh implementable unit and it is the best-aimed one on the
   board** — `cpu<2h`, and it heads the `LT.03`–`LT.07`/`LT.09` chain that is the
   Curiosity-signal seat's arena. `curiosity` is 12 specs, 2 PASS, 0 runnable;
   this is the way back in. Rank it above item 3 if you can only take one.
7. **Still do not pre-empt the Review**: `A4`, `T2.10`'s repair, `SO.07`, `SO.10`,
   `T1.08`'s pipeline repair, `HR.1`'s fixture redesign, `UB.10`'s successor arm,
   and now the two `PS` legibility rows. All the Review's, all dated.

## FOR THE OWNER

**1. NO-DECISION — nothing new is asked of you today, deliberately, for the second
audit running.** Your desk holds four items (`D27`, `D28`, `D29`, `D31`) that this
project's own instrument flags as probably the organs' paperwork — all `costs 0
specs`, all `class: goal`, all `CONDUCT-MISFILED?`. Adding a fifth while that
stands would make the finding worse. Everything below is reported, not asked.

**2. NO-DECISION — your `GOAL.md:187` sentence is half-answered, and the answer it
got back is a real one.** Two of the seven survival primitives you named now have
falsifiable claims, and both were **run the same day they were registered**. The
results, plainly: in W0 **distance genuinely costs** — a scripted traveller pays
monotonically more need-currency over 1.5/3.0/4.5/6.0 m, ~20× the measurable
quantum, and a teleporting twin pays a **dead-flat** bill, so it is the traversal
being charged for and not the clock. **Exertion genuinely costs and is genuinely
repaid** — sustained work halves what the same commands achieve, a fatigue-frozen
twin shows no droop, and rest restores it on the 60 s timescale the spec
registered, which is what distinguishes tiredness from injury. Both specs then
**FAILed**, on one conjunct each: whether Jack can *feel the price coming* varies
by world. That is the ladder doing what you built it to do — the half that was
earned is recorded as earned, and the half that was not is red. Your sentence
stands untouched and unnarrowed. `heavy` and `worth-it` remain, and they are the
builder's to register, not yours to rule on.

**3. NO-DECISION — `D27` decides tomorrow, and this audit produced one clean new
data point for it.** `D27` asks whether this repo should buy mechanical screens,
against a measured 104-of-107 false-positive showing. The `STEERING-DATE-MISMATCH`
screen — reporting-only, unfloored — fired today on the 102nd audit's own prose,
reading a shipped-on date inside a quotation as if it were a deadline. That is a
**false positive produced within six hours of the screen's first exposure to real
text**, and it cost nothing precisely because the screen was shipped unfloored.
Both halves are evidence: the rate is real, and the "unfloored until measured"
discipline `D27`'s own default proposes is what made it harmless. I offer it as a
measurement, not as an argument for either answer.

**4. NO-DECISION — the GPU ration, unchanged since this morning and no longer
worth re-litigating.** Kaggle W37 closes tonight with **27.7837 h unspent** and
`unattributable = 0.0`. **Nothing was charged today** — every unit in this window
was CPU — and that is the correct outcome, not a failure, because no GPU-class
pilot is legally dispatchable while its design answer sits on the Review's desk.
W38 opens tomorrow with 30 h expiring 09-26 and, as things stand, the same
outcome. The composition remains the fact I would want you to hold:
`gpu_hours_no_verdict` totals **48.42 h**, of which **33.78 h is `D1.0` across two
attempts and zero verdicts** — more hours on one undecided arena than the whole
quota we are about to let expire again.

**5. NO-DECISION — the defect class the Review escalated to you this morning at
five occurrences reached SEVEN by lunchtime, and I am completing its report
rather than opening a new one.** The Review wrote: *"if a sixth occurrence follows
a fifth lesson, the conclusion will not be about `ladder_loop.sh` any more."* Six
and seven both followed, within four hours, costing two whole slots. I am still
not asking you to rule, because the diagnosis held up and the builder found and
fixed a genuine engineering defect in its own repair within the hour — with a test
that keeps the broken version as a control. But the shape is now unambiguous and
you should have it in one sentence: **every remedy this class has received in
seven occurrences tells someone about the loss afterwards; none of them can refuse
it.** I have ordered the guard as item 2 to the builder. If an eighth occurrence
happens after a guard exists, that is a different and much more serious
conversation, and it would be yours.
