# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-19 12:37–13:2x UTC — 153rd audit.** First audit slot of the day to
run: `00:37` and `06:37` both read `STOPPED at 90% / 93% weekly usage`. This
slot's own first act, before any reading below, was `review_liveness` failing
and stamping `docs/PROGRESS.md` **STALE** (`bbbe869`, 12:37:03) — so every
number on that page is a record of 2026-10-18 and is quoted here only when I
have re-derived it.

## VERDICT: DRIFTING

**The blackout is over and that is the day's real news. The builder produced
five `rc=0` slots this morning after 31 consecutive dark ones** — 07:07, 08:07,
09:07, 10:07, 12:07, demonstrated `105 → 107` — and it spent them on honest
work: two instrument reds diagnosed and repaired, the regate dirty-tree guard
shipped five days early, the `T2.14 → T2.05` triviality-floor port executed six
days early. §2 is clean: not one threshold moved in the loosening direction in
seven days, no control deleted, no assertion removed, no seed count reduced.
The ledger's content is sound: 107 PASS, every id registered, every recorded
commit resolvable, every PASS that declares a control recorded one.

**What is drifting is that today's repairs cannot land.** The chain is one
causal line and I traced every link at source:

> an `rc=0` slot that stopped mid-wait at 08:36 left the tree dirty → the 08:43
> regate cron bought **five `+dirty` certificates** seven minutes later →
> `unreachable` rose **96 → 101** against a shrink-only floor of **95** → the
> builder correctly diagnosed it, predicted a mechanical heal, and deferred
> `ratchets record` on that prediction → **the heal was refused at 12:43 and
> will be refused by every remaining tick today**, because regate's abstention
> guard is unconditionally closed once the day's CPU spend passes the tightest
> class's slack, which today's own science spend already did.

So the project is carrying a shrink-only ratchet six above its floor, with the
one lane that could clear it locked shut, while the organ that said it would
confirm the heal has ended three of five slots asleep waiting for it.

Ranked by damage to the trustworthiness of the record:

1. **The dirty-stamp heal is REFUSED, not pending** — 64 seconds of CPU
   declined to protect a slack already 597 s overdrawn, holding `unreachable`
   at 101/95 with no legal payer until midnight. §1.
2. **The 2026-10-18 Sunday FULL reported `OVERDUE 6 → 0` while walking past
   eight rows due that same day** — two of them promises it had written to
   itself for that exact sitting, one of which it then walked into and
   reported to the owner as a discovery. §2.
3. **`rc=0` mid-wait recurred three times in five slots**, is the first link
   in finding 1's chain, cost real work twice today, and is still unrouted
   after yesterday's audit named it as the one part of the incident nobody
   owned. §3.
4. **`D41` and `D42` defaults FIRED by this desk.** §6.
5. **The field watch has been dark for two of its last three weekly slots and
   the Review's liveness report says it is live.** `FIELD_WATCH.md` last moved
   **2026-10-05**, fourteen days ago. §4.

---

## §1 — RANK 1: the heal the builder is waiting for was refused, and cannot happen today

### What the lane actually did, from its own log

```
/data/jack-logs/regate.log
2026-10-19T08:52:05  re-bought 8 of 72 cheap certificate(s) ... committed and pushed
2026-10-19T10:43:01  load 15.51 too high — skipping
2026-10-19T12:43:04  stop before T0.21: projected 64s [MEASURED 13.39s x4 + 10s] on
                     top of 4197s already spent today would cross cpu<2h's slack
                     3600s and close that class — declining T0.21, VO.01, DP.00,
                     PG.9, NE.00, T0.28, T0.30, T0.31; still stale next tick
2026-10-19T12:43:04  re-bought 0 of 71 cheap certificate(s); no status moved
```

Those five declined ids — **`VO.01`, `DP.00`, `PG.9`, `NE.00`, `T0.30`** — are
exactly the five `+dirty` certificates the 08:43 sweep bought over the 08:07
slot's uncommitted tree, and exactly the five the builder's 10:07 journal
predicted would *"revert on the stale lane's first clean-tree pass — the five
dirty rows head `regate --list`'s stale lane, MAX_PER_RUN 8 covers them."*

The prediction was reasonable and it is wrong. The lane has had two scheduled
chances since (10:43 refused on tenant load 15.51; 12:43 refused on budget) and
`run status` still lists all five under **DIRTY STAMPS** as I write.

### Why it is a refusal and not a delay — the arithmetic

`scripts/regate.py:228 crossing_a_class_slack()` gates on the **tightest**
class's slack, deliberately, and its docstring gives the reason:

> *"the guard is not this lane's own class budget: it is the TIGHTEST class's
> slack. `used_s > slack_s` is exactly 'that class has >= 1 foreclosed member'
> ... so refusing to be the spender that crosses it means the clerical lane can
> never again be the reason a science class closed."*

The test is `used + est_s > tightest["slack_s"]`. Live:

```
cpu<2h   slack 3600s   spent today 4197s      -> 597s ALREADY OVERDRAWN
                                                 37/60 members unaffordable
```

`used > slack` is **already true**, so `used + est_s > slack` is true for every
`est_s >= 0`. The guard admits nothing — not 64 s, not one second — for the rest
of the UTC day, and today's spend only ever rises. Five more ticks fire today
(14:43, 16:43, 18:43, 20:43, 22:43) and each will print the same line.

**The defect, stated precisely: the guard conflates "do not be the spender who
closes the class" with "do not spend while the class is closed."** Its own
docstring supplies the equivalence that makes the second reading pointless —
`used > slack` *means* the class already has a foreclosed member. Those 37
`cpu<2h` specs are unaffordable with or without the re-buy. The abstention
protects nothing and costs the whole repair a day.

And who overdrew it: **today's own builder slots**, legitimately — `W1.02`'s
re-buy alone billed 1,138.9 core-s. So the science lane forecloses the clerical
lane that repairs the science lane's accident.

### What it costs, measured not asserted

I re-derived the ratchet from the ledger at three revisions, registry constant
at 257 specs throughout:

```
0ba77f0  (10-17, before the Sunday sitting)   unreachable 100 of 257
c350dca  (10-18, Sunday FULL's last commit)   unreachable  98 of 257
HEAD     (now)                                unreachable 101 of 257   floor 95
```

Newly unreachable today: **`DP.04`, `DP.05`, `NE.01`, `PS.06`, `PS.08`**.
Cleared: `T3.06`, `W1.00`. Every one of the five new ones resolves to the same
cause, read off `Ledger.unsatisfied` — *"PASS but stale — DIRTY: ran from a
modified tree at 5f2eb58"* on `DP.00`, `VO.01` or `NE.00`.

So the five dirty certificates ARE the +5, and clearing them returns the counter
to 96. **I checked the builder's diagnosis rather than inheriting it and it is
correct in mechanism**: with `DP.00` clean at `c350dca`, `DP.01`/`DP.02`/`DP.03`
carry roots `['LC.03']` and read `welded<-LC.03`; with `DP.00` `+dirty` they
carry `['DP.00', 'LC.03']`, the `all(root_dead)` quantifier in
`coverage._liveness_state` fails on the live root, and they shed the weld. That
is why `goal_unrunnable` *fell* 7 → 5 and why `coverage` printed a **mandated
irreversible shrink** of the shrink-only `GOAL_UNRUNNABLE_BASELINE`
(*"2 unrunnable-baseline entr(y/ies) are LIVE again and must be removed: DP.02,
DP.03"*). **The builder's refusal to obey it was right** and its LESSONS entry
is the correct generalisation. I looked for a latent quantifier bug behind it
and there is none: a clean PASS is not a root at all, and the one residual case
(`T5.08` behind the foreclosed `T3.06`) is a divergence `champions._unrunnable`
documents as deliberate. **No finding against the instruments here.**

The finding is only this: **a predicted heal is a prediction, not a repair**,
and the 12:07 slot went to sleep waiting on *"the regate-pass watcher ... confirm
the five-certificate heal"* twenty-eight minutes before the 12:43 tick refused
it. Nothing re-reads a prediction.

Residual after the heal: 96 vs floor 95. The last 1 is structural and routed —
96 was the floor from 09-22 to 09-25 and fell to 95 only on `LT.02`'s PASS,
which `lt02-c2-passed-on-the-epsilon-not-on-a-measurement` (OPEN, DUE
2026-10-23) shows was bought on an epsilon. That raise belongs on that row.

### The repair is available today and costs ~206 seconds

**None of the five is a `cpu<2h` spec.** `PG.9` and `NE.00` are `cpu<1min`
(slack 56,548 s, spent 4,197 s); `VO.01`, `DP.00`, `T0.30` are `cpu<10min`
(slack 46,800 s). Both classes read `ok 0/88` and `ok 0/26` unaffordable in
`run status`. The 08:4x timings bound the bill at **72.81 + 107.04 + 2.12 +
4.78 + 19.0 = 205.75 s**. A hand re-buy is legal, affordable and precedented —
the 07:07 slot did exactly that for `T0.02` under the 152nd audit's FTB 1. See
FOR THE BUILDER 1.

---

## §2 — RANK 2: `OVERDUE 6 → 0` was true and the sitting still broke eight promises that night

`review-queue` exits 2 with **8 OVERDUE**, every one `promised 2026-10-18 (1 d
ago)`. `docs/PROGRESS.md`, written by that sitting, carries as its headline
reading:

| `review-queue` | **0 violations**, OVERDUE **6 → 0** | all six disposed this sitting |

Both statements are true. Checked at the revision boundaries:

```
0ba77f0  (last commit before the sitting)   rows carrying DUE: 2026-10-18  = 8
c350dca  (the sitting's final commit)       rows carrying DUE: 2026-10-18  = 8
```

All eight carried that date **before** the sitting opened and still carried it
when it closed. The sitting cleared the six rows whose dates had already passed
and walked past the eight falling due that morning, which went OVERDUE at
midnight. Net disposal of dated promises for the day: **zero**. `review-queue`'s
`IMMINENT` block prints precisely this, before the fact — *"20 live dated rows
fall due on or before the consumer's next cycle ... 14 of them cannot be
discharged by that cycle"* — and the clean number is what reached the page.

**Two of the eight were that sitting's own promises to itself, made for that
exact sitting.** Quoted from the rows:

- `xl00-negative-claim-does-not-assert-its-own-test-could-reject` —
  *"DUE: 2026-10-18 | **THE NEXT SUNDAY FULL SITTING**, and the date is chosen
  by the MODE boundary rather than by the pile. The repair is an ADDED CONJUNCT
  on a passing spec — Part 2 work ... So this desk deliberately did NOT
  implement a strengthening it had already designed and verified as safe.
  2026-10-18 also carries ZERO live rows."* The Sunday FULL arrived, ran a
  Part 2 sweep over sixteen other specs, and did not perform the one Part 2
  strengthening it had deferred into that slot on purpose.

- `steering-page-is-two-days-from-the-launch-cliff-and-31kb-is-provenance` —
  *"DUE: 2026-10-18 | **THIS DESK's execution of THE ARCHIVE** ... if a slot
  opens before 10-18 and this is still undone, **it takes precedence over any
  other act in this file.** That is a priority statement, not a re-date."*
  The sitting instead walked into the cliff this row describes — wrote a `1^23`
  priority block, measured **782 B of margin** below `MAX_ARG_STRLEN`, reverted
  its own edit — and then reported the cliff to the owner as item 6 of FOR THE
  OWNER, citing *"the archival that fixes this is on its own queue row"* as the
  reason not to open a seventh decision. **It cited the row it was breaking, in
  the same paragraph, and the row carried that sitting's own absolute-precedence
  order for that day.**

The other six: `gpu-receipt-head-is-push-time-not-kernel-time` (builder, bill
priced by the desk at `cpu<1min` + 1 slot, *"the hazard is live on every GPU
dispatch until it lands"*); `duration-s-feeds-two-meters`, `t112-pass-stands-on-
t111s`, `bare-git-commit-writes-the-whole-shared-index` (all three dated onto
10-18 by `next_free_due`'s mechanical answer); and `d10-successor-rerun-under-
adopted-gate` + `freeze-release-condition-is-five-specs-deep` — one decision,
honestly stuck behind `T1.08`'s unmoved FAIL, needing a dated re-arm nobody
wrote.

The three honest repairs are ACT, DECLINE, or re-date with a reason. None of the
eight got any of the three. **No instrument can catch this shape**: OVERDUE is
computed after the date passes, so a sitting that disposes yesterday's six and
ignores today's eight scores a clean violation class on the morning it
guarantees eight fresh ones.

**And the desk has not sat today.** `06:37` deferred (`93% weekly usage`),
`09:22` retry `ABORT: load 7.64`, `12:22` retry `ABORT: load 6.44`. Two retries
remain (15:22, 18:22). `review_liveness` reads **STILL STALE**. The organ that
reads this page every morning will not read it until it sits.

---

## §3 — RANK 3: `rc=0` mid-wait, three times in five slots, still unrouted

Yesterday's audit §3 named this as *"the cause of the whole incident, and the
one part of it nobody has routed."* It is still unrouted — I grepped all 184
`ROUTED:` rows — and it fired three more times today:

```
08:07  "Watchers are set — I'll proceed when the completion notification arrives."
08:36  iteration end rc=0 — 107 -> 107
09:07  "W1.02 is still running; I'll resume when the completion notification arrives."
09:21  iteration end rc=0 — 106 -> 107
12:07  "Waiting on the two armed watchers ... confirm the five-certificate heal"
12:35  iteration end rc=0 — 107 -> 107
```

It cost real work twice today, both times verified at source:

- **08:36 → 08:43.** The slot ended `rc=0` with work uncommitted. Seven minutes
  later the regate cron ran over that tree and bought five `+dirty`
  certificates. That is §1's entire chain, and its first link is this class.
- **09:21 → 10:07.** Three clean re-buys (`T0.21` PASS, `T0.28` PASS, `T0.31`
  FAIL→PASS) sat orphaned and uncommitted for 46 minutes until the 10:07 slot
  inherited and committed them at `f729928`. `demonstrated` read **106** at the
  09:07 slot's start — the ladder under-reported itself for an hour.
- **12:35, open now.** The slot went to sleep waiting on the heal that §1 shows
  was refused eight minutes later. Nothing will re-read it.

An `rc=0` that stops halfway is indistinguishable from a completed slot to every
monitor this project owns: the pacer counts it as a non-dark slot, `dark_slots`
reads 0, the ladder log's `iteration end rc=0` is the success line. The only
evidence is prose in the slot summary, which no instrument parses.

---

## §4 — organ liveness, and a false reading in the Review's own report

| organ | today | state |
|---|---|---|
| builder | 07/08/09/10/12:07 all `rc=0`; 11:07 `ABORT load 9.01` | **ALIVE** — blackout broken after 31 dark slots |
| overseer | 00:37, 06:37 `STOPPED at 90%/93%`; 12:37 this audit | first slot of the day |
| Review | 06:37 deferred (usage); 09:22, 12:22 `ABORT load` | **DARK**; 15:22/18:22 remain |
| field watch | 05:37 `STOPPED at 92% weekly usage` | **DARK — second of last three** |

`docs/PROGRESS.md` §2.5 states: *"Field watch live (10-12 Monday, next due
10-19)."* **That is false.** The 10-12 sweep logged `STOPPED at 98% weekly
usage` and produced nothing; `docs/FIELD_WATCH.md` last moved at `32b098c`,
**2026-10-05** — fourteen days ago, not seven. The sitting then used that
reading to decline consuming it: *"FIELD_WATCH.md was **not** consumed (it has
not changed since 10-12 ... so nothing is owed there)."* The conclusion happens
to survive — week-10's findings were genuinely routed on 10-13, and there are no
week-11 or week-12 findings because neither sweep ran — but the project's only
research-ingest organ has been dark for two of its last three weekly slots and
no page says so. Re-derive an organ's liveness from its log, never from the
previous report's table.

**Builder productivity, counted off `/data/jack-logs/ladder.log` for the 24 h to
12:37:** 5 iterations, **5 `rc=0`**, PASS delta **105 → 107** (`T0.28` and
`T0.22` repaired from the 04:57 regate reds; `T0.31` FAIL→PASS inherited). Also
7 slots `STOPPED at 90–93%` overnight and 5 `ABORT: load` — the usage week
released itself by the calendar at 04:59Z, by no act of anyone's, exactly as the
10-13 re-dates predicted.

---

## §5 — compute honesty

**GPU.** `Budget._week()` reads **2026-W42**; kaggle `remaining_range` =
**(30.0, 30.0)** — the full free allocation, **zero drawn**, expiring Saturday
**2026-10-24**. The last charged job in `gpu_budget.json` is
`jack-ladder-1791461828`, **2026-10-08** (W40): **eleven days with no GPU spend
at all.** W41's 30 h expired unbought. This is the **seventh consecutive week**
in that state and the reason is not the pacer:

```
coverage QUEUE DEPTH
  gpu<20min   0  EMPTY    <- NOT FILLABLE: pilot BLOCKED on evidence (SM.03); repair is a REDESIGN
  gpu<2h      1  UB.10    (no FRESH dispatch here) — UB.10 is VOID, an arm to repair
  gpu<8h      0  EMPTY    <- NOT FILLABLE: pilot BLOCKED on evidence (LC.07); repair is a REDESIGN
```

Every GPU class is empty or holds only a VOID. **The hours are not being wasted
by a loop that will not run; they are unspendable because no designed buyer
exists.** `D24` (the `LC.07` scale-transfer arena: bought, shrunk, or declared
unaffordable) is the decision that would create one and its `decide_by`
2026-09-11 is 38 days past. Stated as `D40` requires: these hours would likely
have expired regardless.

**GPU-hours with no verdict: 51.47 h total**, unchanged — `D1.0` 33.78 h across
2 attempts for **0 verdicts**, `UB.10` 0.30 h / 0 verdicts, 6.32 h across 21
`UNATTRIBUTED` jobs (ratchet at its floor of 21). No new unattributed spend.

**CPU.** 4,197 s of 57,600 s used today. `cpu<2h` is **597 s past its own
3,600 s slack**, 37 of 60 members unaffordable until midnight — which is §1's
lock. `cpu<1min` and `cpu<10min` have 13–15 h of slack each.

---

## §6 — decisions: two defaults fired, and six conduct entries nobody has executed

### `D41` and `D42` — FIRED BY THIS DESK, appended to `docs/DECISIONS_NEEDED.md`

Both are `class: goal`, armed, `decide_by 2026-10-18`; `decisions --check`
listed both as the only `OVERDUE — DEFAULT IS DUE TO FIRE` entries; today is the
first legal day. **The owner did not rule by 2026-10-18, so the pre-registered
defaults fired.** Both are monotone and both are, exactly, the status quo. The
full blocks with the firing conditions, the evidence I re-derived at source, the
price each carries and how to reverse them are in `DECISIONS_NEEDED.md`; the
transcription to `DECISIONS_RESOLVED.md` is FOR THE BUILDER 4.

I re-verified `D41`'s measurement rather than inheriting it: `action_training_loss`
has **zero call sites outside `experiments/`** (every caller is a spec test);
`make_action_optimizer` has **four spec callers and one script probe, no pipeline
caller**; `TrainingPipeline.py:493` is still `AdamW(weight_decay=1e-4,
eps=1e-5)` against the shared recipe's plain `Adam`. `T1.11` is FAIL.

For `D42` I re-verified `claim_dead = 3` and found **one material change the
entry does not know about**: its named downstream home,
`five-commitments-are-claim-dead-behind-foreclosures`, is now **DECLINED** (it
was OPEN, DUE 2026-10-14 when the entry was written). So after this firing
nothing anywhere holds the successor-spec question except a closed register
entry — the `D30`/`D40` disease by name. Flagged in FOR THE OWNER 2.

### Six `CONDUCT-DESK` entries are stale, two of them by nearly a month

`decisions --check` prints these as *"desk-executable, not the owner's —
execute it, report it, do not ask"*:

```
D33  due 2026-09-23  STALE by 26 d      D40  due 2026-10-10  STALE by  9 d
D35  due 2026-09-24  STALE by 25 d      D46  due 2026-10-17  STALE by  2 d
D38  due 2026-10-04  STALE by 15 d      (D47 2026-10-24, D48 11-01, D49 10-27 live)
D39  due 2026-10-15  STALE by  4 d
```

All six are Review-authored and the Review is the desk that can execute them.
Part of the cause is `D46`'s own subject — the usage gate taking out the Review
— and the Review has sat 3 times in the last 6 days. But a conduct entry is
listed *"so a stale conduct entry cannot silently self-approve,"* and six of
them have now been listed daily without moving. Not mine to execute; named here
because the page that would act on them is the page that reads this one.

**Clean elsewhere.** `MEANS-ESCALATED`: **none** — no fork a measurement could
settle is on the owner's desk. `UNDECLARED`: **0**, so there is nothing to arm
this audit and the ratchet has not grown. `firing transcriptions: 32 of 32`.
`ratchet ok (0/0 undeclared, 0/0 unrouted-owner-ask, 0/0 vanished-owner-ask,
0/0 default-action-expired, 0/0 firing-diff)`. I read `PROGRESS.md`'s FOR THE
OWNER section and all six items route to existing `D` ids (`D46`, `D48`, `D47`,
`D30`, `D46`) — `decisions` matches on all five and none has vanished.

---

## §7 — the mandated first four, and bakeoff hygiene

**`coverage`** exits 2. `0 commitments with NO declared spec` — the 2026-08-10
hole stays closed. Standing reds, all previously measured and owned, none new
today: `claim_dead = 3` (smell, shelter/building, thermal — `D42`, now fired);
`NO-LIVE-PATH` 6 distinct holes; `park_release_pairs` 3; `PASS-ON-DEAD-
DEPENDENCY` **6 vs floor 3**, unchanged since 2026-10-17 and carrying its
read-not-reasoned note (`T0.13`/`T1.11`/`T1.08`/`T6.03`). `unreachable` is §1.
`GOAL.md` citations: 16 cited, **0 dangling**; 5 unrunnable, of which the 4 NEW
(`GEN.02/03/06/09`, all `welded<-LC.07`) are owned by
`gen-four-revival-needs-an-affordable-lc07-successor` (OPEN, DUE 2026-10-24).

**`champions`** exits 0, every ratchet at its floor, **every seat says what
would unseat it** (`UNDECLARED 0/0`). 10 standing violations, all previously
reported: `VERDICT-IS-A-VOID` + `TRIGGER-UNREACHABLE` on Learning core (held BY
VERDICT off `LC.03` = VOID); `ARENA-UNREACHABLE` + `TRIGGER-UNREACHABLE` on
Fast/slow coupling; `VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED` on World;
`NO-ARENA` on ASR and Speaker ID; `UNCONTESTED` on Vision encoder and the
plastic-only decree. **§7's answer on bakeoff hygiene is this line**: the
strongest marking in the file is still backed by a VOID — `D10` seated
wm-latent BY VERDICT off `LC.03`'s VOID on 2026-09-01, with a single-arm caveat
on its face. That is a decision taken off a non-verdict and it is recorded as
such in `champions`' ratchet (`2/2 unverified verdicts`, at floor). Nothing new
entered `DECISIONS_RESOLVED.md` since; no winner was chosen inside a noise
margin this week, because no bakeoff ran this week.

**`review-queue`** exits 2 — §2. Also standing and not re-routed: drain **85
cycles** for 73 live rows at the measured net rate (`D47`); 20 IMMINENT against
a measured 6/cycle with 12 on today alone; 7 rows DELIVERED-AWAITING-STAMP.

**Ledger integrity (§1 of the mandate), checked row by row.** 107 PASS; every
id resolves in `BY_ID`; every recorded `commit` resolves in git except the five
bearing the `+dirty` suffix, whose base sha `5f2eb58` exists (the suffix is the
stamp, not a missing object); **zero** PASS rows declare a control and record no
`control_metrics`; the only two PASSes without a control, `T0.01` and `T0.10`,
both carry an explicit `NoControlByDecision` naming the 52nd audit's B5. **No
finding.**

**Thresholds and controls over seven days (§2 of the mandate).** Five commits
touched `registry*.py` or `experiments/tests/`. Every numeric change is an
addition or a tightening: `b6069dc` ports `T2.14`'s `TRIVIAL_FLOOR = 0.05` and
`floor_selftest_ok` into `T2.05` **unchanged**, adding two VOID gates with a
known-answer self-test while `WM_RATIO` stays 0.8, `CTRL_TOL` stays 0.98 and
the claim conjunct is byte-untouched; `1ee9c76` adopts `T0.32`'s
`rtf.require_feasible` as a VOID lane in three specs and re-declares `W1.02`
`cpu<2h → cpu<10min` with the arithmetic in the commit (a narrower class, i.e.
stricter). No threshold moved down, no control deleted or weakened, no `_check`
gained an `or`, no seed count reduced, no assertion removed. **No finding — and
I state that plainly rather than manufacturing one.**

---

## §8 — the honest summary

**Are we closer to a curious humanoid that climbs the ladder?** Today, barely —
but for the first time in a fortnight the answer is not simply no.

`demonstrated` moved `105 → 107` and both points are Tier-0 instrument
self-tests re-bought after the regate lane correctly broke them. That is the
measuring apparatus repairing itself, not Jack learning anything. Of the five
slots, **one unit was about the creature**: the `T2.14 → T2.05` port, and it is
the best unit on the board — zero GPU hours, zero world edits, no threshold
moved, and it converts the project's longest unattributable FAIL into an
attributable one. Everything else — the regate guard, the WAITS-ON readings, the
two instrument repairs, the `W1.02` re-buy — was apparatus.

Against `GOAL.md`: smell, shelter and *too cold kills him* still have **no live
falsifier**, nineteen days after `coverage` first said so, and `D42`'s default
has now fired to say we are leaving them that way. Curiosity's deciding spec
`LT.03` is VOID. The unified-brain seat is `UNDECIDED` with a VOID arena. Six
weeks of free GPU have expired into cost classes with nothing to put in them,
and the decision that would create a buyer (`D24`) is 38 days past its date.
The ladder is the right ladder — `0 commitments uncovered` — but three of the
owner's own commitments are rungs with nothing on them.

The honest shape of the day: **the organs are healthy, awake, and spending
almost all of their recovered capacity on each other.** One slot stopped
halfway, a cron bought five dirty certificates through the gap, a shrink-only
ratchet went six above its floor, the lane that could clear it is locked by the
day's own science spend, and the next slot went to sleep waiting for a heal that
had already been refused. Every link in that chain is a correct local decision.
That is the thing to be worried about — not that anyone was dishonest, but that
five correct decisions composed into a day where 206 seconds of CPU cannot be
spent to clear a red that every instrument prints.

---

## FOR THE BUILDER

Ordered cheapest and most perishable first. Items 1 and 2 expire at midnight
UTC.

1. **Re-buy the five `+dirty` certificates BY HAND this slot — do not wait for
   regate.** `VO.01`, `DP.00`, `PG.9`, `NE.00`, `T0.30`, from the now-clean
   tree. **The heal you predicted at 10:07 was REFUSED at 12:43** and will be
   refused by every remaining tick today (14:43, 16:43, 18:43, 20:43, 22:43):
   `crossing_a_class_slack` tests `used + est > tightest_slack`, and `cpu<2h` is
   already 597 s past its 3,600 s slack, so the test is unconditionally true for
   any estimate. Read the line yourself at `/data/jack-logs/regate.log`
   `12:43:04` — it names all five. **The hand path is legal and affordable:**
   none of the five is a `cpu<2h` spec (`PG.9`/`NE.00` are `cpu<1min`,
   `VO.01`/`DP.00`/`T0.30` are `cpu<10min`, both classes `ok` with 13–15 h of
   slack), the bill is **~206 s** off the 08:4x timings, and the 07:07 slot
   already did exactly this for `T0.02`. **Exit condition, so this is
   falsifiable:** `run status` DIRTY STAMPS drops from 7 to 2 (`T6.03` and
   `PL.02` are not payable), `unreachable` returns to **96**, `goal_unrunnable`
   returns to **7**, and `coverage` stops mandating the shrink of
   `GOAL_UNRUNNABLE_BASELINE`. **Then run `run ratchets record`** — it is
   deferred on a prediction that is now known false, and quote the slot line
   before recording.

2. **Fix the guard, in the same slot if the bill allows: `scripts/regate.py:228
   crossing_a_class_slack` must not refuse when the line is already crossed.**
   The guard exists so the clerical lane *"can never again be the reason a
   science class closed"* — and its own docstring supplies the equivalence that
   makes today's refusal pointless: `used_s > slack_s` **is** "that class
   already has a foreclosed member." Once that holds, abstaining protects
   nothing: the 37 `cpu<2h` specs are unaffordable with or without the re-buy,
   and the lane's 64 s can neither close nor un-close the class. The minimal
   change that keeps the original intent: refuse only when the spend would
   **cross** the line (`used <= slack < used + est`), and when `used > slack`
   already, fall back to the candidate's **own** class slack — which is the
   quantity that actually governs whether this particular spec is affordable.
   Do not simply delete the guard; the 128th audit's RANK 2a is a real scar
   (4,218 s spent on day one, 37 specs foreclosed). Price the staleness bill
   before you edit — `regate.py` is not in any spec's `IMPL_DEPS` as far as I
   can see, but verify with `run stale-cost scripts/regate.py`.

3. **`longrun_unbound` / `T0.32` is DUE TODAY and untouched.** It is item 4 of
   `PROGRESS.md`'s FOR THE BUILDER, it still holds `T0.32` at FAIL over ten
   specs, and the `t032` stamp on 10-18 did not pre-empt it. Three other rows
   also fall due today: `cpu-class-ladder-has-no-rung-between-2h-and-48h`
   (which is the structural half of item 2 above), `seven-instrument-readers-
   are-gated-by-no-spec`, and `salt-differential-has-no-coverage-counter`.
   Twelve rows share today's date against a measured capacity of 6.

4. **Transcribe `D41` and `D42` to `docs/DECISIONS_RESOLVED.md`.** Both fired
   this audit; the blocks are appended to `DECISIONS_NEEDED.md` under
   `## D41 — RESOLVED BY ARMED DEFAULT` and `## D42 — RESOLVED BY ARMED
   DEFAULT`. `decisions --check` currently reads `32 of 32` transcribed; it will
   read `32 of 34` until you do. Same-day transcription is the precedent set on
   `D43`.

5. **`gpu-receipt-head-is-push-time-not-kernel-time` is OVERDUE (promised
   2026-10-18) and is yours.** The desk discharged the design half and priced
   the bill at `cpu<1min` + one slot (`T0.12`'s re-buy). Its own routing says
   *"the hazard is live on every GPU dispatch until it lands"* — and W42 holds
   30 free GPU-hours expiring Saturday. The repair is `ref` resolving to a
   commit rather than the literal `"main"` in `repo_preamble` and `build_job`
   (`experiments/gpu.py:210`, `:332`), plus the five call sites taking the
   default. If you cannot reach it today, that is a legitimate answer — but it
   is the Review's row and only the Review may re-date it, so say so in the
   journal rather than letting a second day pass silently.

6. **Route the `rc=0`-mid-wait class.** Yesterday's audit named it unrouted; it
   has now fired three more times and cost two measurable losses today (the
   08:36 gap that produced the five dirty certificates, and the 09:21 gap that
   orphaned three re-buys for 46 minutes). You may not write the Review's row,
   but you can route it: a slot that ends `rc=0` while a declared watcher is
   still armed should either commit its work first or exit non-zero, and
   `dark_slots` reading 0 through three of these is the detector's blind spot.
   One line of evidence to attach: `demonstrated` read **106** at the 09:07
   slot's start because the 09:07 slot's own predecessor had not committed.

---

## FOR THE OWNER

**1. `D41` AND `D42` FIRED THIS MORNING. You did not rule by 2026-10-18, so the
pre-registered defaults fired, and both are reversible by one sentence from you
at any time.**

- **`D41`** (which artefact is the thing we are building) fired to **(iii)
  NEITHER — hold both artefacts, repair each instance on its own row, re-ask in
  writing.** Nothing was repaired cosmetically. `T1.11` stays an honest FAIL and
  was **not** given a call site. I re-verified the measurement at source today:
  `action_training_loss` still has zero callers outside `experiments/`;
  `make_action_optimizer` still has four spec callers and no pipeline caller;
  the two recipes still differ (`AdamW`/`wd 1e-4`/clip 1.0 in the pipeline vs
  plain `Adam`/no decay/clip 2.0 in the shared recipe). **The price, restated
  because a firing may not drop it:** the drift stays live, a third bridge may
  be built and go uncalled the same way, and `demonstrated 107` keeps being
  reported without the caveat that it describes one of two artefacts. **To
  reverse: name (i) or (ii) in one sentence.**

- **`D42`** (which of three claim-dead commitments matters most) fired to **(iv)
  HOLD — no successor is designed, and the three commitments stay claim-dead and
  said so.** Nothing was unparked; every park keeps its pre-registered rule;
  `coverage` keeps printing `claim_dead = 3`. The desk's own recommendation —
  **name `thermal ("too cold kills him")` as the one** — was deliberately NOT
  taken, because choosing among your commitments would be a desk writing
  `GOAL.md`. **One thing has changed since the entry was written and it makes
  the price worse than the entry states:** the queue row named as this
  question's live home, `five-commitments-are-claim-dead-behind-foreclosures`,
  is now **DECLINED** — terminal. So as of this firing, **smell, shelter and
  "too cold kills him" have no live falsifier, no open queue row, and no open
  register entry. The question now exists only inside a closed entry**, which is
  exactly the failure `D30` and `D40` were raised about. **To reverse: name one
  commitment in one sentence and a successor spec gets registered that
  iteration.**

**2. The one thing I would ask you to read if you read nothing else.** Your own
words are the capability target: *every sense a human has, one brain, learning
by living.* Three of the senses and needs you named by hand have been
unfalsifiable for nineteen days. The parking was honest every time — `SM.02`'s
pre-registered both-fail branch fired, `SH.01` returned `ORACLE_CANNOT` with the
hut's direction in the agent's own observation. Nobody cheated. But a commitment
with no live claim spec is invisible to every instrument this project owns, and
as of this morning it is invisible to the register too. **One sentence from you
— "thermal first" — turns the largest hole in the ladder back into work.**

**3. `D49` is now in a state where the register's own advice cannot be
followed, and it will self-decline on 2026-10-28.** The Review routed it
yesterday as `class: constitution`; `constitution` is not a legal register class
and `T0.28` went red on it at 04:57, so the builder amended the field to
`conduct` to clear the gate, disclosing the change in full and touching nothing
else. The pick is defensible — `decisions`' own heuristic says a question about
how the organs work belongs in `conduct`. But `conduct` carries the instruction
*"desk-executable, not the owner's — execute it, report it, do not ask,"* and
`D49`'s entire content is a proof that **the set of actors permitted to
discharge it is empty**: every repair on its row is a loosening, `1^13` forbids
the builder, strengthen-only forbids the Review. So the tool now tells the desk
not to ask about a row the desk may not act on, and the default fires on 10-28
to **DECLINE a measured defect on `PL.02`** — `GOAL.md`'s only registered
falsifier for your 2026-08-09 plastic-only decree. The Review recommends against
its own default and so do I. **The ask is one sentence: does the
strengthen-only law admit a narrow, audited exception for an arm measured to
quantify over its own nulls?** Nothing else on this page needs you.

**4. `D30`'s required pairing, counted this audit and not inherited.** **The
blackout is OVER.** The builder produced five `rc=0` slots this morning after 31
consecutive dark ones; `dark_slots` reads 0. It ended by the calendar — the
usage week reset at 04:59Z — and by no act of yours or anyone's, which is the
third time the mechanism has released itself while `D30`, `D40` and
`docs/PROGRESS.md` all told you that resuming required a file written by your
hand. The gate that acted overnight was the **90% hard stop**, not the pace
line: seven slots logged `STOPPED at 90–93% weekly usage` between 00:07 and
06:07, and `week:all models` read **18%** at the 10:07 slot. `week:Fable` read
**7%** and is **not** the gate. **`2026-W42` holds 30.0 free Kaggle GPU-hours,
0.000 drawn, expiring Saturday 2026-10-24 — and every GPU cost class reads EMPTY
or NOT FILLABLE.** This is the seventh consecutive week. Said with the honesty
`D40` requires: **the hours are not the argument.** The argument is that `D24` —
whether `LC.07`'s scale-transfer arena is bought, shrunk or declared
unaffordable — is **38 days past its `decide_by`** and it is the decision that
would create a buyer. Until it is answered, free GPU weeks will keep expiring
and the pacer will keep being blamed for it.

**5. NO-DECISION, reported because the previous sitting's report of it is
wrong.** Your research-ingest organ is dark. `field_watch` logged `STOPPED at
92% weekly usage` this morning and `STOPPED at 98%` on 10-12; `docs/FIELD_WATCH.md`
last moved **2026-10-05**. `docs/PROGRESS.md` says *"Field watch live (10-12
Monday, next due 10-19)"* — it was not live on 10-12. Nothing is owed from the
sweeps that did not run, so no finding is lost; but two of the last three
weekly windows on the outside world were spent by another tenant's usage, and
the organ has no retry poll of its own the way the Review does. Routed nowhere
new by design: `D46` already carries the usage gate's reach across organs, and
this is one more organ inside its scope. If you want the field watch given
retry polls like `review.sh --retry`, say so and it will be routed.
