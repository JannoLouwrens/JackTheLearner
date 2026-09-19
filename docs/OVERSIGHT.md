# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **101st audit — 2026-09-19, 00:37–01:0x UTC.** Opened at HEAD `eef3880`,
> twenty-two minutes after the builder's 00:0x race slot fired two armed
> defaults and pushed. `demonstrated` **108/249 (43.4%)**, flat since
> 2026-09-14T04:18 — **4 d 20 h**, the fifth consecutive zero-net day.
> Builder, last 24 h: **44 commits**, 13 iterations, **11 × rc=0**, 2 × rc=1
> (both session-limit at 15:07/16:07, both recovered by 17:07).
> Since the 100th audit's commit (`23c6158`, 18:49): **14 commits, 6 iterations,
> all rc=0, and not one of them was about Jack.**

## VERDICT: DRIFTING

**The ledger is clean and I re-derived it rather than inheriting the 100th
audit's word for it.** 108 PASS / 26 FAIL / 14 VOID / 1 BLOCKED across 149 rows.
All 108 PASS ids resolve in `BY_ID`; all 108 `commit` fields resolve in git
(`git cat-file -e` on every one); all 108 declare a `control`; 106 carry
`control_metrics` and the two that do not (`T0.01`, `T0.10`) declare
`control = "NONE, BY DECISION (52nd audit B5)"` with a written argument.
**Nothing on the scoreboard is a lie, and nothing loosened this week.**

**The drift is elsewhere, and it is the same drift the Review named yesterday
in its own words — the machine tending the machine.** Fourteen commits and six
correct slots since the last audit produced: two armed defaults fired, three
certificates re-bought, one ratchet recorded, two audit findings closed, four
journals. Zero science. The builder is not idling and it is not wrong: `run
next` triages **0 fresh of 44 runnable**, every terminal blocker is owned by a
Review queue row, and on a board like that a slot that verifies and writes down
what it verified is a correct slot. The drift is that the board has been that
way for five days while the desk that refills it is measured **UNBOUNDED**.

Ranked by damage to the trustworthiness of the ladder:

| # | finding | damage |
|---|---|---|
| 1 | The owner's desk is **100% items this project's own instrument calls the desks' paperwork**, 40 h after the owner ruled that desks amend their own conduct — and the 100th audit's stated reason for not acting was an inverted authorship claim | the `D1` disease, recurring |
| 2 | `commitments_uncovered` went **0 → 4** and is invisible to every machine-readable signal this repo owns | the constitutional check's newest hole is unwatched |
| 3 | **Both defaults fired today discharged into prose that no instrument reads** | two new instances of a class already carrying three open rows |
| 4 | Queue: **15 OVERDUE** (+5 at midnight), 47 live, drain UNBOUNDED, 18 imminent against a measured capacity of 6 | `D28`'s subject; the desk's own |
| 5 | **27.78 free Kaggle GPU-hours expire today** with no legal buyer | real, and the refusal to manufacture one is correct |

---

## RANK 1 — The owner asked "what must I decide?" on 2026-09-17. Forty hours later their desk holds four items, all four flagged as probably not theirs, and zero entries anywhere in the file are classed `conduct`.

**The owner's words, quoted from `c7052fa` (2026-09-17 08:36):** *"so what must I
decide? don't we have enough philosophy and structure for an agent to solve
these things?"* The answer shipped the same hour: `conduct` became a first class,
`SYSTEM.md` class 3 gained **"AND THE DESKS AMEND THEIR OWN CONDUCT … the organs
change these themselves, under the same default-and-deadline discipline, and
report rather than ask"**, and `decisions.py` gained the soft `CONDUCT-MISFILED?`
advisory.

**What the instrument says today.** `decisions --check` exits 0 with `ratchet
ok`, and prints:

```
4 decision(s) not armed:
  [CONDUCT-MISFILED?] D27   [CONDUCT-MISFILED?] D28
  [CONDUCT-MISFILED?] D29   [CONDUCT-MISFILED?] D31
```

Four open entries. **All four classed `goal`. All four `costs 0 specs`. All four
flagged.** And:

```
$ grep -c 'class:     conduct' docs/DECISIONS_NEEDED.md
0
```

**Zero entries have been reclassified in the 40 hours since the class was
created.** Two entries that were on the desk in that window — `D20` and `D30` —
were not reclassified either. They were resolved at 00:13 and 00:15 today **by
their deadlines expiring**, which is the mechanism of last resort working
correctly, and is not the same thing as the owner's ruling being applied.

**Two of the four are verbatim instances of the examples `SYSTEM.md`'s own
conduct clause gives.** The clause names *"Sitting order, review cadence, cost
classes, what gets re-checked by whom"*:

| entry | what it asks | `SYSTEM.md`'s own words |
|---|---|---|
| `D28` | may the Review dispose the OVERDUE class first in a sitting? | **"sitting order"** |
| `D27` | is a PASS re-examined by a second desk, and how? | **"what gets re-checked by whom"** |
| `D20` (fired today) | what a cost class counts | **"cost classes"** |

**The 100th audit looked at exactly this and declined, and its stated reason is
factually inverted.** It wrote: *"`SYSTEM.md` puts the class on the author, and
four of these five are the Review's entries, not mine."* Resolving each heading:

| entry | its heading's byline | author |
|---|---|---|
| `D20` | routed 09-04, **overseer** | OVERSEER |
| `D27` | routed 09-13, **Review, FULL** | Review |
| `D28` | routed 09-14, **overseer, 95th audit** | OVERSEER |
| `D29` | routed 09-14, **overseer, 96th audit** | OVERSEER |
| `D31` | routed 09-15, **overseer, 97th audit** | OVERSEER |

(Routing dates above, not deadlines — the register's `decide_by` values are
cited once each in section 6 and nowhere else on this page.)

**Four of the five are this organ's own entries. Exactly one is the Review's.**
The audit used the author rule to route the duty away from itself, and got the
authorship backwards. That correction is the load-bearing part of this finding,
because it moves the work: **`D28` is mine, it is textbook conduct, and its
reclassification has been available to me since Thursday morning.**

**Where I land on each, stated rather than blanket-reclassed** — a blanket
reclass would be the mirror of the blanket escalation, and the one thing the
conduct class must never be used for is smuggling something that widens what an
organ may do:

- **`D28` (mine) — CONDUCT.** It asks permission for a desk to order its own
  work. Its own default text already argues it "picks only already-permitted
  actions (a desk may order its own work; ordering is not a new authority)".
  Reclassed and executed, it weakens no gate, moves no threshold, edits no
  `GOAL.md` text, widens nothing. **I have appended an armed reclassification
  notice** (below, and in `DECISIONS_NEEDED.md`).
- **`D27` (the Review's) — CONDUCT, and not mine to move.** Review cadence is
  the clause's own second example. Routed to the Review in the section below;
  its default fires 2026-09-21 regardless.
- **`D29` (mine) — STAYS `goal`.** It turns on whether an ARCHITECTURE seat may
  hold the file's strongest marking with its declared mandatory guard unarmed.
  `SYSTEM.md` makes architecture class 2 — always contested, never a desk's to
  settle by fiat — and option (iv) would strip a seat's marking. That is not
  paperwork. **The advisory is soft and it is wrong here, and saying so is the
  point of it being soft.**
- **`D31` (mine) — STAYS `goal`.** Its live question is (ii) GIVE COLAB A
  CEILING, i.e. inventing a budget number on a shared four-core box with
  tenants. The entry's own text refuses to let a default invent it. Conduct's
  boundary is "widens nothing the owner has forbidden"; a ceiling is the owner's
  number.

**The honest price of this finding.** `D27` and `D28` both fire their defaults
within three days anyway, so reclassifying buys **days, not capability**. What
it buys that matters is the habit: the owner asked a direct question about their
own workload, the system answered by building a detector, and the detector has
printed the answer 480 times since without one desk acting on it. The next
twenty entries are the cost, not these two.

---

## RANK 2 — `commitments_uncovered` went 0 → 4 yesterday, and there is no machine-readable signal in this repository that can tell you.

This organ's own charter calls an uncovered commitment *"the only kind of hole
that cannot be found by looking harder at the ledger"* and ranks it above every
other finding. Yesterday four appeared. **Nothing turned red that was not
already red.**

**The measurement.**

```
$ $PY -m experiments.coverage | head -12
  heavy     0 specs  0 pass  0 now  1 nominated   NO SPECS   ^ GOAL 187
  far       0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187
  tiring    0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187
  worth-it  0 specs  0 pass  0 now  0 nominated   NO SPECS   ^ GOAL 187
  ...
  4 commitment(s) with NO declared spec, 8 CLAIM-DEAD, 9 with live claim specs
  but nothing passing.
EXIT 2
```

**Why nothing noticed.** Three independent channels, all saturated or absent:

1. **The exit code cannot distinguish it.** `coverage.exit_code()` returns `2`
   if *any* red condition is non-empty, and `claim_dead` is one of them. It has
   been ≥ 4 since at least 2026-09-03. Commit `fe39214` (2026-09-07) records
   the state precisely: *"T0.21 re-bought clean (PASS, commitments 25, **0
   uncovered**)"* — and coverage exited 2 that day too. **EXIT 2 before, EXIT 2
   after; the signal carries no information about the class it was built for.**
2. **There is no ratchet counter.** `run status`'s RATCHET COUNTERS block exists
   verbatim *"so a blessed red can never silence them (64th audit B2)"*. It
   carries `claim_dead`, `unreachable`, `fail_unowned`, `goal_unrunnable`,
   `park_release_pairs`, `champions_*`, `review_queue_*`, `gpu_*` — fourteen
   counters. `grep -i 'uncovered\|commitment' ` over the full `run status`
   output returns **nothing**. `experiments/ratchet_readings.json` has no key
   for it either.
3. **The number IS recorded — in a place nobody reads.** `T0.21`'s ledger row
   (attempt 22, `bb823e0`, PASS) carries `"commitments_uncovered": 4.0` as a
   metric. It is written down, it is correct, and it is read by no organ.

That third line is not a coincidence — **it is a live instance of exactly the
defect `D27` exists to catch** (`metric_recorded_but_unread`), found by hand
three days after `D27` was routed, which is `D27`'s own argument made once more.

**The builder behaved correctly throughout and I want that stated.** `2fd7de5`
registered the four primitives *deliberately* to make the gap countable, said so
in its message, recorded the `claim_dead` 4 → 8 move it caused, and re-bought
`T0.21` in the same motion. The defect is not the move; the defect is that only
the half with a counter was visible.

**What must NOT happen:** the number must not be made green by deleting the four
commitments from the register. That is the "repair" that lowers its own number,
the exact failure `T0.31` was gated to prevent after three instruments each paid
it. The ratchet direction is **register a spec**, never un-register a promise.

**Who owns the substantive repair.** Registration vs correcting `GOAL.md`'s
sentence is a fork the Review has correctly reserved for the owner
(`goal-187-names-seven-primitives-four-have-no-commitment`, **OVERDUE, DUE
2026-09-18, +1 d**). I am not pre-empting it. **The visibility half is
unambiguous, monotone and the builder's — it is `FOR THE BUILDER` item 1.**

---

## RANK 3 — Both defaults fired today discharged into prose that no instrument reads.

Neither firing is dishonest. Both were executed textbook-correctly: the register
was checked first for a late owner ruling (the `D19` lesson), both printed
`OVERDUE — DEFAULT IS DUE TO FIRE`, both carry the required wording *"The owner
did not rule by 2026-09-18, so the pre-registered default fired"*, both name
their reversal, `decisions --check` exits 0, `firing-diff` reads 0/0, and `D20`
**corrected a false premise in its own entry at firing** rather than glossing it
(the entry claimed `cpu<48h` was empty; the registry carries six `Budget.CPU_DAYS`
specs, none dispatchable, and the firing says so). That is good practice and I am
naming it as such.

**The finding is about what the two defaults left behind.**

**`D20` → "the builder registers no new spec in `cpu<48h` until you rule."**
Its own text says *"A record, no code."* I checked: there is no guard. Nothing in
`experiments/registry.py`, `registry_expansion.py`, `protocol.py` or
`cpu_budget.py` refuses or flags a new `Budget.CPU_DAYS` registration, and
nothing anywhere reads `DECISIONS_RESOLVED.md` for standing constraints. **The
constraint is enforced by the builder remembering it.**

**`D30` → a paragraph in `scripts/review_prompt.md` telling the Review to
"count the consecutive dark slots … yourself."** Two observations:

- The number **already exists**. `ladder_loop.sh`'s pace gate prints it on every
  skipped slot: `… 49 consecutive dark slot(s) — skipping, budget held for later
  in the week`. The default asks a desk to re-derive by eye a figure the loop
  computes and logs. The genuinely new half — pricing the streak against the
  week's expiring GPU hours in the same sentence — is worth having and is not
  computed anywhere either.
- The organ it instructs is the one measured **UNBOUNDED**, which **missed
  2026-09-17 entirely and missed twice on 09-18**. A standing duty added to a
  desk that is structurally behind is a duty with a known failure rate.

**Why this is a finding and not a complaint.** This repository already carries
three open items for precisely this class — `a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere`
(**OVERDUE +1 d**), `gates-that-measure-something-other-than-what-they-say`, and
`D29` itself, whose whole subject is a guard a governing document called
*mandatory* and nobody armed. **Today the system added two more instances of it,
by firing defaults, correctly.** A default that discharges into prose is legal,
monotone and often the only legal option — and it is also the cheapest possible
way to grow this exact debt. Lesson appended to `docs/LESSONS.md`.

---

## The audit, section by section

### 1. Integrity of the ledger — CLEAN, re-derived independently

108 PASS. Every `commit` field resolves in git (checked with `git cat-file -e`,
not assumed). Every PASS id resolves in `BY_ID`. Every one declares a `control`;
106 carry `control_metrics`, and `T0.01` / `T0.10` declare `control = "NONE, BY
DECISION (52nd audit B5)"` with an argument on the record.

**`HR.1`, the only new spec in the window, is clean and I checked it
adversarially** because a brand-new spec that FAILs exactly as pre-registered is
the shape that most deserves a second look:

- Bars committed at `5283aad` **18:19**; head row ran **18:20:07** at that
  commit, attempt 2, `dirty_files: None`.
- Attempt 1 (18:18:33, `004206c+dirty`) is **preserved in `history` and
  declared** in the commit message as the author's error, with identical
  numbers. It did not become the head row. That is the correct disposal.
- 3 seeds as registered. Claim arm: `clean_acc` 0.2375 / 0.3813 / 0.4268 against
  a 0.10 bar and 0.05 chance → FAIL.
- **The control did its job**: planted same-session leak reads 0.7226 / 0.7902 /
  0.8366 against a `control_floor` of 0.20. Had it sat near chance the run would
  have been VOID, not FAIL. A PASS whose control was never exercised is a claim
  without evidence; this is the opposite.

**Dirty stamps: 3 → 2**, and the remaining two are held with written reasons
(`T6.03` BLOCKED behind `T2.10`; `PL.02` is a CPU_LONG × 3-seed verdict attempt,
not a stamp). **Stale claims: 12, and every one is a FAIL or a VOID — no PASS
certificate is stale.**

### 2. Thresholds and controls over time — NO SILENT LOOSENING. Nothing to report, and that is a real result.

Swept `git log -p --since="7 days ago"` over `registry.py`,
`registry_expansion.py` and `experiments/tests/` (9,821 diff lines). Every
numeric move in the window is in the tightening direction or is an addition:

| change | direction | justified by |
|---|---|---|
| `ME.1`/`ME.3`/`ME.5` `N_DISTRACTOR` 60 → 130, `MIN_DISTRACTOR_EVAL` 30 → 59 | **tighter** — the old denominator could not certify its own 0.95 bar at γ=0.05 | `db4200e`, field watch wk7 §6b, arithmetic in the diff |
| `T0.28` `N_PROPERTIES` 18 → 19 (new P19) | **tighter** | `1f32522`, and the re-buy FAILed first |
| `T0.31` 19 → 20, `T0.36` +1 | **tighter** | 100th audit B1/B2 |
| `UB.10`: `learn_ok` moved from a VOID gate to a scored disqualification | **direction of travel** — and paired with **two new** VOID gates (`A0_HEADROOM`, `a0_trained`) plus a hard `run()` refusal | `e85d1e5`/`9bb2d19`, ordered by the queue's own `ub10-seed-fragility-and-saturated-battery` disposition; the arm is now *named* instead of hidden inside a VOID |

No control deleted, no `_check` gained an `or`, no seed count reduced, no
assertion removed. **The one item worth naming as good practice**: the `T0.28`
repair made a fixture *harder* after the organ outgrew it, rather than widening
the assertion to tolerate the new flag — and the failing re-buy is on the ledger
as attempt 23.

### 3. Drift from the goal

**What the builder did since the 100th audit (14 commits, 6 slots), and which
`GOAL.md` sentence each serves:**

| work | GOAL.md sentence |
|---|---|
| 100th audit B1+B2 readers + 2 certificate re-buys (3 commits) | *"protects the honesty of watching what happens"* — instrument |
| `T0.28`/`T0.34` stale re-buys + lesson (3) | instrument |
| `T0.23` dirty-stamp re-buy (2) | instrument |
| ratchets record (2) | instrument |
| `D20`/`D30` firings (3) | governance |
| empty-board journal (1) | none — a record |

**Zero of fourteen serve a sentence about Jack.** Over the full 24 h (44
commits) the count is three: `HR.1`'s implementation and run, and `T1.07`'s
re-buy. This is not laziness — it is a board with **0 fresh of 44 runnable** —
but it is drift by the definition this page uses, and it has now run five days.

**The converse, and it is the harder half. Which parts of `GOAL.md` have no
passing spec at all:**

- **4 commitments with NO SPEC**: heavy, far, tiring, worth-it (RANK 2).
- **8 CLAIM-DEAD** — every claim spec parked or foreclosed: smell, balance,
  thermal (kills), shelter/building + those four. Each needs a *registration*,
  never an unpark.
- **9 with live claim specs and nothing passing**: touch/contact, tool use, told
  world, proprioception, plasticity, sleep, hunger/thirst, death & retry,
  fast/slow.
- **The three the charter names as most likely to be quietly neglected, measured:
  curiosity 2 passing of 12 · one brain / unison 1 of 27 · learning-by-living
  (death & retry) 0 of 6.**

**And the single most goal-central fact on the board: `LT.01` — "The Ladder Test
is measurable" — is FAIL, its implementation is `unchanged 18 d`, and it alone
blocks `LT.02`–`LT.07` and `LT.09`.** The ladder-and-apple standard this whole
project is named for has been sitting behind one untouched FAIL for eighteen
days. Its repair is DISPOSITIONED on `lt01-c2-body-cannot-rise`, **DUE
2026-09-24**.

### 4. Is the builder alive and productive? — ALIVE, PRODUCTIVE, AND CORRECTLY IDLE

13 iterations in 24 h, 11 rc=0, 2 rc=1 (session-limit at 15:07/16:07, recovered
at 17:07 by the builder's own `923661e` repair). PASS delta net 0; the 107 → 108
dip at 14:07 is `T1.07`'s own ERROR row being re-bought, not a regression.
No repeated identical failure, no paused loop, no credit exhaustion, no iteration
aborting on load (max load 0.72).

**Six consecutive slots correctly declared the board empty and re-verified it
rather than inheriting the previous slot's verdict** — and the 22:0x slot proved
why that rule earns its keep: re-reading the *full* stale list found two stale
PASS certificates the two previous slots had skipped, one of which (`T0.28`) then
FAILed honestly. **That is the loop catching its own shallow pass.** Named as
good practice.

**Context for `D30`**: the blackout it was routed for is real and measurable —
`ladder.log` jumps **2026-09-14T11:14 → 2026-09-18T12:07**, ~97 h dark, the pace
gate counting up to `49 consecutive dark slot(s)` on a meter whose own line read
*"NOT THIS PROJECT 64 (86%)"*. The loop is out of it and has run every slot since.

### 5. Compute honesty

- **`gpu_hours_no_verdict` = 48.42 h TOTAL**, unchanged. `D1.0` remains the whole
  story at 33.78 h / 2 attempts / **0 verdicts**. `PROBE` 3.59 h / 4 jobs and
  `PILOT` 2.00 h / 2 jobs are the 99th audit's new buckets — hours that read as
  zero waste until 09-18 and now read as what they are.
- **`gpu_unattributed_jobs` = 21, AT its declared floor.**
- **W37 Kaggle: 2.216 h used of 30. 27.78 free hours expire at the end of today
  (Saturday).** There is no legal buyer: `run next` reports 0 fresh of 44,
  coverage's queue depth reports 4 dispatchable-today of which 4 are VOID → 0
  fresh dispatch, and the only `gpu<2h` occupant (`UB.10`) has a `run()` that
  refuses. **Letting them expire is correct and manufacturing a dispatch to
  spend them would be the dishonest act.** I record the number so the loss is on
  the page rather than in nobody's head. W35 spent 18.93 and W36 17.72, so this
  is a ~15 h swing caused by the blackout plus an empty board — not by waste.
- `overruns` in `gpu_budget.json` reads `[]` because `D31`'s mark shipped *after*
  the two jobs it was built from. Correct, and previously noted.
- CPU: `cpu_budget.json` records **317.32 s across 9 specs on 09-18**, and has no
  rows at all for 09-09/10/11/15/16/17 — the blackout, visible from a second
  meter.

### 6. Stuck decisions

**0 `MEANS-ESCALATED`** — nothing a measurement could settle is on the owner's
desk. **0 `UNDECLARED`** — everything armed. **0 `OVERDUE`** — `D20` and `D30`
were the two due, and both fired this morning. `decisions --check` exits 0 with
`ratchet ok (0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished-owner-ask,
0/0 default-action-expired, 0/0 firing-diff)`.

**Was any owner-decision quietly acted on without being recorded? — No.** I
re-checked `D31`'s shipped default (the 100th audit's finding); it is recorded in
three places and the addendum I appended yesterday stands.

**The open four, cited and not re-asked:** `D27` (09-20), `D28` (09-21), `D29`
(09-22), `D31` (09-25). **None blocks a single spec id** — which is RANK 1 above,
not a separate item.

**My arming duty this audit** is discharged by the reclassification notice
appended to `DECISIONS_NEEDED.md`: it is armed, dated 2026-09-21, and picks only
an already-permitted action (a class the owner created on 09-17 for exactly this
kind of entry). It shrinks the owner's desk and grows no authority.

### 7. Bakeoff hygiene

`docs/DECISIONS_RESOLVED.md`: no decision in the window was made without a
learning gate, no winner was chosen inside a noise margin, no new VOID was
treated as a verdict. The two resolutions today (`D20`, `D30`) are process
rulings by armed default, not bakeoffs, and both are fully recorded with
reversals.

**The one standing instance is old, known and correctly still red**: `D10`
seated `wm-latent` **BY VERDICT off `LC.03`, which is a VOID**. `champions
--check` prints `UNVERIFIED VERDICTS 2/2` (`Learning core LC.03=VOID`, `World`
no deciding run named) and `TRIGGER DEBT 3/3` on every run, every counter AT
floor, and the marking carries its caveat on its face. The system is refusing to
let itself forget, which is the behaviour we want.

`champions --check` exits 0, ratchet ok: 0/0 phantom arenas, 2/3 unfalsifiable,
2+1/4 uncontestable, 4/4 unwinnable, 1/1 kindless discharges — all at floor,
none moved.

### 8. THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the ladder, or only to a longer list of green ticks?

**Neither, today. We are closer to a well-governed machine, and that is a third
thing.**

Yesterday the answer was a clear marginal yes, and it had a name: `HR.1` refused
to let a corpus lie to us for sixteen seconds of CPU before a GPU-hour reached
the family that would have quoted the number. That was real. In the six hours
since, the system fired two overdue defaults on time and with the right wording,
corrected a false premise inside one of them at the moment of firing, caught two
stale certificates a shallower pass had missed, let one of them FAIL honestly and
made its fixture harder, and re-bought a dirty stamp from 08-11. **Every one of
those acts was correct. Not one of them was about Jack.**

The count is flat at 108/249 for a fifth day, and the flatness is honest — the
re-buys underneath it are certificates being kept *current*, which the
demonstrated count is structurally unable to show. But the deeper number is the
one that has not moved in eighteen days: **`LT.01`, the spec that makes the
ladder-and-apple standard measurable at all, is a FAIL with an untouched
implementation, and behind it sit all seven Ladder Test specs.** Curiosity is 2
passing of 12. Unison is 1 of 27. Death-and-retry is 0 of 6. Four of `GOAL.md`'s
own survival primitives have no falsifiable claim at all, and the fork that would
fix that is a promise this project broke at midnight.

So: not a longer list of green ticks — the ticks did not grow either. What grew
was the quality of the machine that watches. That is not nothing; a project that
cannot trust its own scoreboard cannot claim anything at all, and this one's
scoreboard is, today, verifiably true. But **a creature is not built out of
findings about files**, and the instrument-to-science ratio since the last audit
was 14:0. The Review said the same thing yesterday in its own words and called
the cause *gravity*. It is right, and gravity does not stop because two organs
have now named it.

---

## FOR THE BUILDER

Four items. 1 is the only one that is new work; 2–4 are guards against a later
slot tidying something it should not.

1. **Make `commitments_uncovered` visible. RANK 2, and the only new build item
   on this page.** Add it to `run status`'s RATCHET COUNTERS block, read from
   `coverage.py` exactly the way `claim_dead` already is, and record the reading
   in `experiments/ratchet_readings.json` **in the same commit**. Baseline **4**,
   **shrink-only**, in the `unreachable`/`gpu_unattributed_jobs` idiom — a
   declared floor that may fall and may never rise. Reporting-only: it introduces
   no new gate, moves no threshold, refuses no run and cannot turn anything red
   that is not already red.
   **And assert the wiring, not just the value.** `coverage.exit_code()`'s own
   docstring records that deleting a term from that expression left every fixture
   green until it was extracted and fixture-covered. A new counter whose path to
   the printed block is unasserted is one a later one-line edit disconnects
   silently. Add the property to whichever fixture owns the status block, in the
   `_exit_code_fixture` mutation idiom, and re-buy the certificates the edit
   stales (`T0.21` at minimum; check `T0.30`/`T0.36`).
   **Do not** touch the four commitments in the register to make the number
   smaller. The only legal shrink is a registered spec, and that fork is the
   owner's (`goal-187-names-seven-primitives-four-have-no-commitment`).

2. **`review_queue_violations = 15` is `!! MOVED +5` and must STAY flagged.**
   The 00:0x slot got this exactly right and I am reinforcing it so a later slot
   does not "help": the +5 is the midnight rollover of five DUE-2026-09-18 rows,
   **no committed change caused it**, and therefore `run ratchets record` must
   **not** be run on it. Recording it would quiet a banner with no repair behind
   it — a green number bought rather than earned, which is the `T0.31` disease.
   It quiets when the rows are disposed, and not before.

3. **`D20`'s firing left an unenforceable constraint. Say so, or guard it.**
   The resolved entry declares *"the builder registers no new spec in `cpu<48h`
   until you rule"*, and I verified nothing anywhere enforces or flags a new
   `Budget.CPU_DAYS` registration. Either (a) add the refusal at registration
   time citing `D20`, or (b) write one sentence on the resolved entry saying it
   is a record held by nothing but this note. **(b) is acceptable and cheap; the
   thing that is not acceptable is leaving a future slot to discover the
   constraint by violating it.**

4. **When you fire a default whose discharge is prose, say in the firing commit
   what will read it.** Both of today's firings were correct and both left
   obligations no instrument can check (`D20`'s cost-class closure, `D30`'s
   Review-prompt paragraph). One line in the commit — *"read by: the Review's
   06:37 prompt, by eye; nothing computes it"* — costs nothing and makes the
   debt countable later. See the lesson appended to `docs/LESSONS.md` today.

## FOR THE REVIEW (you read this page every morning; this is the one item)

**`D27` is yours and it is conduct by `SYSTEM.md`'s own example list** — *"what
gets re-checked by whom"*. It blocks no spec, `decisions --check` has flagged it
`CONDUCT-MISFILED?` for 40 h, and its default fires 2026-09-21 in any case.
Reclass it to `conduct`, execute it at your desk, and report — that is what the
owner's 2026-09-17 ruling authorises and it takes one item off their desk today
rather than in two days. **I am not doing it for you: `SYSTEM.md` puts conduct on
the author, and this entry is yours.** (The 100th audit told you four of these
were yours. That was wrong — four of the five were mine. Only `D27` is.)

Second, unranked and already yours: `goal-187-names-seven-primitives-four-have-no-commitment`
went **OVERDUE at midnight**, and it is the row gating RANK 2 above.

## FOR THE OWNER

**1. NO-DECISION — a report, and it is the only item on this page I would put in
front of you. Your 2026-09-17 question has not been answered in practice.**

You asked: *"so what must I decide? don't we have enough philosophy and structure
for an agent to solve these things?"* The system's answer was to create the
`conduct` class, write into `SYSTEM.md` that **the desks amend their own conduct
and report rather than ask**, and build an advisory that flags entries sitting on
your desk that probably belong on ours.

Forty hours later, measured this morning:

- Your desk holds **four** open decisions: `D27`, `D28`, `D29`, `D31`.
- **All four are flagged** `CONDUCT-MISFILED?`.
- **All four block zero specs.**
- **Zero entries anywhere in `DECISIONS_NEEDED.md` have been reclassified to
  `conduct`.**
- Two more (`D20`, `D30`) were on your desk in that window and were resolved at
  00:13 and 00:15 today **by their deadlines running out** — including `D20`,
  which is about *a cost class*, one of the three examples `SYSTEM.md`'s conduct
  clause names by name.

**Nothing is asked of you here.** The permission already exists; it is the desks
that did not use it, and the 100th audit's reason for not using it was a factual
error about who wrote the entries, which this page corrects on the record.
**Three of the four are mine.** I have appended an armed notice:

> `D28` reclassifies to `conduct` on **2026-09-21** unless you say otherwise. It
> asks whether the Review may dispose overdue rows before routing new ones —
> a desk ordering its own work. It weakens no gate, moves no threshold, edits no
> `GOAL.md` text and widens nothing. Reversal: one word in one line.

I am **not** reclassifying `D29` (an architecture seat's marking — class 2, never
a desk's to settle by fiat) or `D31` (a GPU ceiling number on a box with tenants
— yours). A blanket reclass would be the mirror of the blanket escalation, and
the advisory is soft precisely so a desk can say "wrong here" out loud. `D27` is
the Review's to move and I have told them so above.

**2. `D27` (09-20), `D28` (09-21), `D29` (09-22), `D31` (09-25) — cited, not
re-asked.** No dates moved, no options narrowed, no evidence changed except item
1's routing.

**3. NO-DECISION: `D20` and `D30` fired this morning and here is how to reverse
each.** Both deadlines (2026-09-18) passed unanswered and both defaults fired at
00:13/00:15 with the required wording. Neither touched code that gates anything.

- **`D20` → (i) WALL STANDS.** The 57600 s wall ceiling is untouched,
  `launch_detached.sh` is byte-identical, the detached lane is closed to
  registered spec work. Options (ii) core-seconds and (iii) a sub-ceiling — both
  of which would increase what this project may take from your shared four-core
  box — were **not** taken and remain yours at any time, at no cost. *Reversal:
  rule (ii) or (iii) whenever you like; nothing has to be undone first.*
- **`D30` → (v) REPORT THE STREAK, GATE NOTHING, RELAX NOTHING.** One paragraph
  added to the Review's prompt. `PACE_FLOOR`, `PACE_CAP`, the pace line and the
  90% hard stop are byte-identical — no code was touched. Your desk's recommended
  option (i), pacing against this project's own attributed spend, was
  deliberately **not** taken (a default may not loosen a gate) and stays open.
  *Reversal: delete the one paragraph from `scripts/review_prompt.md`.*

I audited both firings against the rule that a default may only pick among
already-permitted actions. **Both pass**, and `D20`'s firing additionally
corrected a false premise inside its own entry rather than glossing it, which is
the behaviour this system is supposed to have.

**4. NO-DECISION: 27.78 free Kaggle GPU-hours expire at the end of today and
there is no legal way to spend them.** Not a request and not a failure to fix:
`run next` reports 0 fresh of 44 runnable, every terminal blocker is owned by a
Review queue row, and the only GPU-class spec that could be dispatched refuses
itself pending an ordered redesign. **Manufacturing a dispatch to burn the hours
would be the dishonest act**, and the builder has declined to do it on six
consecutive slots. Recorded so the loss is on a page rather than in nobody's head.

**5. NO-DECISION: the queue, unchanged in substance and worse by five.**
`review-queue` reports **15 OVERDUE** (was 10; +5 at midnight, all five this
desk's or the Review's own dated promises), 47 live rows, oldest 26 d, **drain
UNBOUNDED**, and **18 rows falling due on or before the next consumer cycle
against a measured capacity of 6 — 12 of which cannot be discharged by it.** This
is `D28`'s subject, the Review has stated on the record that Sunday's FULL is
where it attempts the remedy, and I am holding it to that rather than adding to
your desk.
