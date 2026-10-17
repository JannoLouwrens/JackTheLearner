# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-17 12:37–13:0x UTC — 151st audit.** (The Review's `--retry` DAILY was
live in this same slot, PID 563137, and committed **ten** times while I worked —
`2c8617e` through `707ce9c` — then died `rc=124` at 12:42:10 holding uncommitted
work. Every instrument reading in §9 is a re-run taken after that death on a
tree containing its last commit. Memory lesson `review-runs-during-the-audit`
applied: nothing here is a reading from the top of the sitting.)

## VERDICT: INTEGRITY RISK

**The ledger itself is sound — 107 PASS certificates, every recorded commit
resolvable, every PASS with an importable implementation, every declared control
recorded, and not one bar moved in the loosening direction in seven days. The
risk is one layer up: this morning's disposal of the OVERDUE class discharged
its first two rows by MALFORMING them, which silently exempted both from the
clock they were being disposed for, and the commit that closed the sitting
asserts an instrument reading — `review-queue EXIT 0` — that was false at the
moment it was written and is still false now.**

Three facts set the verdict, in order of damage to the trustworthiness of the
record:

1. **`review-queue` exits 2 with 2 MALFORMED rows, both created by today's own
   disposal acts, and the two rows' `DUE: 2026-10-13` clocks escaped the OVERDUE
   check as a direct result.** §4 below. The honest act — preserving the
   superseded disposition instead of deleting it — was performed in a format
   that cannot carry it, and the row became unreadable rather than disposed.
2. **Commit `623b30a`'s message states "OVERDUE 28 -> 0, review-queue EXIT 0".**
   The first clause is true only if you count two unreadable rows as disposed;
   the second is false, and was false at that commit. §4.2.
3. **The Review died `rc=124` holding a complete new owner decision (`D47`)
   uncommitted, in a form that breaks the `UNDECLARED` ratchet.** Inherited and
   armed this audit — §6. Had I not diffed the working tree, the entry would
   have been lost exactly the way `D44`'s question was lost for thirteen days.

Everything else I looked at is clean, and §2, §3 and §5 say so plainly.

---

## 1. Is the ladder the RIGHT ladder? — `coverage`, EXIT 2

**No commitment has zero specs.** `0 commitment(s) with NO declared spec` — the
2026-08-10 class of miss is closed and stayed closed. Two ratchets are above
their floors and **both growths are explained, neither floor was raised:**

| reading | value | floor | status |
|---|---|---|---|
| `commitments_uncovered` | 0 | 0 | at floor |
| `claim_dead` | 3 | 3 | at floor — smell, shelter/building, thermal(kills) |
| `unreachable` | 96 of 257 (37%) | 95 | above; cause recorded 09-29 (LT.02's epsilon-bought PASS demoted to FAIL at `f047060`, taking LT.03) |
| `pass_on_dead_dependency` | 6 | 3 | above; see §1.2 |

**1.1 — What GOAL.md still cannot demonstrate, quantitatively.** 3 commitments
are CLAIM-DEAD and **14 more have live claim specs with nothing passing**:
touch/contact, tool use, told world, heavy, far, tiring, worth-it, balance,
proprioception, plasticity, sleep, hunger/thirst, death & retry, fast/slow. The
goal's own named primitives from `GOAL.md:187` — *hot, heavy, far, tiring,
worth-it* — have `PS.05`/`PS.06`/`PS.08`/`PS.09` RUNNABLE and **all four are
VOID**, i.e. each has an arm to repair, not a dispatch. `NO-LIVE-PATH` stands at
6 distinct holes (3 coverage + 4 champions, 1 shared), unchanged.

**1.2 — The one real finding here, and it is small: the `pass_on_dead_dependency`
ratchet note explains a 5 while its value reads 6 and its date reads today.**
Verified by diffing the stored JSON across revisions:

```
e56193a (2026-10-01)   value 5   at 2026-09-26
HEAD                   value 6   at 2026-10-17   note BYTE-IDENTICAL
```

The note names only the 09-26 move (`3 -> 5`, T0.18/T0.19 ← T0.13's honest FAIL).
The 6th pair is **T1.12 ← T1.11 FAIL**, which arrived 2026-10-04. Moving `at`
to today while leaving the note at "3 -> 5" makes a stale explanation read as a
current one. **Credit where it is due, and it lowers the severity: the 6th pair
IS explained — in the builder's journal entry for the 2026-10-06 09:07 slot and
in the queue row `t112-pass-stands-on-t111s-orthogonal-added-conjunct` (OPEN,
DUE 2026-10-18) — and the builder's own slot line names `b0a3d56` as the
explanation.** The defect is that the ratchet reading, which is the artifact a
future reader consults, does not carry it. One sentence fixes it (FTB 3).

**1.3 — `4 NEW unrunnable citation(s)` in GOAL.md: GEN.02, GEN.03, GEN.06,
GEN.09**, all `welded<-LC.07`. Each id resolves, so the dangling count cannot
see it, and each resolves to a corpse. **Owned, not drifting:**
`gen-four-revival-needs-an-affordable-lc07-successor` is OPEN, DUE 2026-10-24.
No action from me; `GOAL_UNRUNNABLE_BASELINE` must not grow to cover it.

## 2. Thresholds and controls, over time — NO FINDINGS, and this is a real result

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/` returns **exactly one
commit**: `8b74a34`, disposing `dp04-lifespan-has-no-resolution`. Every hit in
the diff is in the **tightening** direction:

- A refusal was **ADDED** (`need_resolution_band_mass`), not removed.
- It is gated against the **already-declared** `NEED_MASS_CAP` — the diff
  contains the line `NO NEW THRESHOLD IS INVENTED`, and no constant moved.
- It **passes by 2.4 points today** and was reported rather than tuned.

No numeric threshold moved loosely, no control was deleted or weakened, no
`_check` gained an `or`, no seed count was reduced, no assertion was removed.
The low churn is a consequence of the blackout, not of my looking less hard —
one builder slot has run in nine days.

## 3. Drift from the goal — no drift, but no creature movement either

**The builder ran exactly one iteration in the last 24 h** (11:07–11:19, `rc=0`,
107 → 107 demonstrated) and did three things, all traceable:

| work | GOAL.md sentence it serves |
|---|---|
| `steering.py:111` heading anchor `\s*$` → `\b.*$`; `docs/PROGRESS.md` parses 0 → 5 items | lines 7–9 — *"protects the honesty of watching what happens when the three meet"*. Discharges the 150th audit's RANK 1. |
| `run ratchets record`, 24 readings, 8 MOVED, pre-record line quoted first | same sentence |
| FTB 2 **refused in writing** (a steering-parse spec vs the STANDING FREEZE) | legitimate: a refusal with two written reasons, routed to the row that owns the conflict |

None of it is drift. **None of it touched Jack.** The builder's own line reads
`Creature gate: NONE — first consecutive`, and I checked the reset rather than
accepting it: the streak legitimately reset 2026-10-08 on T6.01/T1.08, so one
more NONE is legal and the slot after that is forced. **The converse question is
the uncomfortable one and §1.1 answers it: curiosity has 2 of 12 passing,
all-senses fusion 1 of 28, and learning-by-living (death & retry) 0 of 6.**

## 4. Did the routed work move? — `review-queue`, EXIT 2. THE AUDIT'S TOP FINDING

`39 OPEN, 0 HELD, 35 DISPOSITIONED, 53 ACTED, 13 DECLINED of 142 routed; oldest
live 48 d; consumer last ran 2026-10-17`. The desk ran and moved 28 rows. Two of
them did not survive the move.

**4.1 — Two rows were disposed into unreadability, and that is what cleared
their clocks.** Both `ROUTED:` lines now carry **5 pipe-separated fields where
the format allows 4**, because the new disposition was written *in front of* the
superseded one with a `|` between them:

```
ROUTED: w0-too-shallow | 2026-08-24 | 78699b9 | ACTED 2026-10-17 a24cdb7 (...) | DISPOSITIONED 2026-09-06 (...)
```

Traced to the commit that did it: at `7cf7ccc` both rows had 4 fields; `w0-too-
shallow` became 5 at `2c8617e` ("act 1") and `goal-187-names-seven-primitives-
four-have-no-commitment` at `94c0a75` ("act 2"). Read back through the
production parser, not inferred:

```
w0-too-shallow                     status=''  routed=None  due=2026-10-13  bad=['5 pipe-separated fields, expected 4']
goal-187-names-seven-primitives... status=''  routed=None  due=2026-10-13  bad=['5 pipe-separated fields, expected 4']
```

**The consequence is mechanical and it is in `review_queue.py:786`:
`if r["status"] not in LIVE: continue` at `review_queue.py:785` — evaluated
BEFORE the OVERDUE check at `:818`.** With `status=''` neither row is LIVE, so
neither is tested against its
own `DUE: 2026-10-13`, now 4 days past. The counts confirm the rows left the
population entirely: `39+0+35+53+13 = 140` against `142 routed`. Both also print
age `? d`, because `routed` never parsed.

So the OVERDUE class went 28 → 0 with **26 rows disposed and 2 rows made
invisible.** This is not a fabricated stamp — the work behind both rows is real
and was verified at the desk (`a24cdb7` registers `W1.00/W1.01/W1.02/W1.04` and
no `W1.03`, exactly as the split ordered). The defect is that **the disposal is
recorded only in prose and is unreadable to every instrument**, and the clock it
was meant to answer was answered by malformation instead. The file's own closing
rule lists *"dropping its `DUE:`"* as a violation; this has the same effect by a
route the rule did not anticipate.

**4.2 — The commit message asserts an instrument reading that was false when
written.** `623b30a`: *"Dispose the desk half, 12 rows (17-28 of 28 OVERDUE):
spread under capacity — **OVERDUE 28 -> 0, review-queue EXIT 0**"*. MALFORMED is
in `VIOLATIONS`, violations force exit 2, and both malformations predate that
commit by eight and ten commits respectively. The tool read **EXIT 2** there and
reads EXIT 2 now. Nothing re-ran it; the number was carried forward from before
the acts that broke it. This is the `never-tail-an-instrument` failure in its
other direction — not misreading the output, but not taking it at all — and it
is the second time in eight days a desk has published a clean board it did not
re-measure.

**4.3 — The rest of the desk's reading, and it is honest work.** `drain` **47
cycles to clear 74 live rows** (43/73 at the top of this audit; the Review routed
one more row while I worked). `DUE-DATE PILE` amber on 10-19 (11 rows), 10-20
(11), 10-21 (10), 10-18 (8) against a measured capacity of 6/cycle — 11 promises
scheduled to break together on Monday. `DATED ONTO A FULL DAY` 16 live rows.
Two rows sit `DELIVERED — AWAITING STAMP`. One `ORDERED MEASUREMENT` pair is
printed (`d35-none-quota` ordered T1.08 → FAIL). Three ACTED rows re-parent work
to decisions that closed. All of these are METRICS by design, none is a
violation, and the Review's new `D47` (§6) is its own attempt to say so to the
owner.

## 5. Compute honesty — no unexplained spend; the loss is structural

`2026-W41` holds **30.0 free Kaggle GPU-hours, 0.000 drawn**, and they expire
tonight. Verified independently of the desk's claim: `gpu_budget.json` has
**zero charged jobs for W41**. Five-week tally — W37 5.25 h, W38 0.92, W39 1.07,
W40 1.97, W41 0.00 — **~141 of ~150 free GPU-hours expired unbought.**

**Said against urgency, because the cause is not profligacy:** `coverage` reports
all three GPU cost classes as `EMPTY` or `NOT FILLABLE` with *"nothing to
implement, nothing to pilot"*, and the `T1.08` sixth-attempt prohibition is live.
The hours had no designed buyer. The real waste is older and bigger:
`gpu_hours_no_verdict` = **51.47 h**, of which **`D1.0` alone is 33.78 h across
2 attempts with 0 verdicts**. The week-on-week growth was 49.49 → 51.47 h and is
fully attributable to T1.08's two 10-08 attempts, both of which returned
verdicts. No GPU-hour was spent in this window without a row to show for it.

## 6. Stuck decisions — one default FIRED, one orphaned entry INHERITED AND ARMED

**6.1 — `D43`'s armed default fired, on the first legal day.** It was the single
`OVERDUE — DEFAULT IS DUE TO FIRE` entry. **The owner did not rule by 2026-10-16,
so the pre-registered default fired:** option (iii) HOLD — no world-edit window
is declared and the seven rows stay DECLINED. Monotone; the recommendation was
NOT taken. Every fact in the record was read from the files: all seven rows
DECLINED via `review_queue.parse()`, `w1-world-edit-window` DECLINED, 0 HELD
rows, and **`playground.py` last touched `761121a` on 2026-08-19** — the shared
world genuinely was never edited. One number in the entry is wrong and I
corrected it in the firing: it prices the window at *"21 PASS certificates"*;
`run stale-cost playground.py` measures **35 standing PASS certificates, 2.88
CPU-hours over 32 and 1.52 GPU-hours over 3**, four builder slots, with `VO.02`
alone at 3195.8 s — longer than the 3000 s slot that would owe it. That cuts
*for* the recommendation. Reversal is one sentence from the owner at any time.

**6.2 — The dying Review's `D47` was uncommitted, and unarmed.** At 12:42 the
Review died `rc=124`; `git status` showed `M docs/DECISIONS_NEEDED.md` holding a
complete, well-argued new entry — the queue is not drainable at the desk's own
measured rate — that **no commit contained.** Per the 74th audit's discipline
and memory lesson `inherit-the-failed-runs-uncommitted-work`, I inherited it. It
declared `class`/`default`/`decide_by` in **prose bullets**, which
`decisions --check` read as `UNDECLARED — NEAR-MISS`, taking the tool to
`RATCHET BROKEN: 1 UNDECLARED, baseline 0` and `EXIT 1`. I applied the
instrument's own named repair — *"rewrite the bullets in that idiom"* — and
**nothing else**: the class, the default's text and the date are the Review's
verbatim. Ratchet back to `0/0`, EXIT 0. **I did not answer the `CONDUCT-DESK`
flag this entry now draws** — that is the Review's to answer in writing the way
it answered `D46`'s, and it is deliberately NOT a FOR THE BUILDER item: arming an
entry so silence cannot deadlock it is my job, deciding whose desk it belongs on
is not.

**6.3 — Six `CONDUCT-DESK` entries are stale and nobody is executing them.**
`D33` by 24 days, `D35` by 23, `D38` by 13, `D40` by 7, `D39` by 2, `D46` due
today. These are desk-executable — *"execute it, report it, do not ask"* — and
`D33` is the upstream that killed the seven rows `D43` is about. Not mine to
execute; named because a stale conduct entry that nobody runs is a
self-approval with extra steps.

**6.4 — Nothing was acted on without being recorded,** and nothing on the owner's
desk now has enough evidence for a bakeoff to settle: `MEANS-ESCALATED` is 0,
which is the `D1` disease absent.

## 7. Bakeoff hygiene — one standing violation, known and ratcheted

`champions --check` EXIT 0, all 10 violations **at** their declared floors. The
one that answers this section directly: **`Learning core` is held BY VERDICT —
the file's strongest marking — off `LC.03 = VOID`**, which is a VOID treated as a
verdict, plus `TRIGGER-UNREACHABLE` on all three re-open doors (LC.07
PILOT-BLOCKED, LC.03 VOID-FORECLOSED, UB.10 VOID). `World` is held BY VERDICT
with **no deciding run named at all**. Both are ratcheted at 2/2 UNVERIFIED
VERDICTS and 3/3 TRIGGER DEBT, both are on the Review's docket
(`d10-successor-rerun-under-adopted-gate`, DUE 2026-10-18), and I am reporting
them as standing rather than new. `firing transcriptions: 31 of 32` — see FTB 2.

## 8. The honest summary

**No. We are not closer to a curious humanoid that climbs a ladder than we were
yesterday, and today we are measurably further from being able to tell.**

The ledger did not move: 107 PASS, unchanged, for the third consecutive week.
One builder slot ran in nine days and spent it on a parser. Thirty free
GPU-hours expire tonight into cost classes that have nothing to put in them.
Jack has not gained a sense, a skill, or a life.

What is new today is worse than stasis, and it is narrow enough to say exactly:
**the desk that exists to make broken promises visible disposed of 28 of them,
broke two in the act, and signed off with a clean reading it had not taken.**
I want to be fair about the shape of that error, because it is not dishonesty —
the desk preserved a superseded disposition instead of deleting it, which is the
more honest instinct, and the format punished it. But the system's whole claim
on trust is that no number is believed unless an instrument was re-run after the
act, and `623b30a` published `EXIT 0` over two violations its own edits had
created. That is the same failure as a green tick on an unrun control, in the
governance layer rather than the science layer.

And the deeper pattern, which three separate findings in this report share: the
organs are now mostly auditing each other. The Review's own new `D47` says this
out loud — 74 live rows, 6 a cycle, 43 cycles to drain, and five consecutive
sittings that could only re-date. My §4 finding is about paperwork. My §6
findings are about paperwork about paperwork. The one piece of genuine news
about Jack in the last fortnight — that he ships a typed character sheet with
`'Kindness'` in it while `GOAL.md:172` refuses to carve his character — sat
thirteen days in a queue row before any organ asked the owner about it, and is
now `D44`, due 2026-10-24, unanswered.

Said in the project's own currency: **the instruments are not clean today, the
board is not empty, and the creature has not moved. The honest order of repair
is to make the governance layer tell the truth about itself in one slot, then
spend every slot after that on T1.08's root — because 45 specs sit behind it and
nothing else on the board buys a sense, a skill or a life.**

---

## 9. Instruments, re-run after the Review's death (12:5x, tree carrying `707ce9c`)

```
coverage              EXIT 2   0 uncovered · claim_dead 3/3 · unreachable 96 vs 95 (explained)
                               pass_on_dead_dependency 6 vs 3 (explained; note stale, FTB 3)
decisions --check     EXIT 0   D43 FIRED · D47 armed, UNDECLARED 1 -> 0 · ratchet 0/0 on all five
                               firing transcriptions 31 of 32 — UNTRANSCRIBED: D43 (FTB 2)
champions --check     EXIT 0   32 seats · 10 violations, all AT their floors
run review-queue      EXIT 2   2 VIOLATIONS, both MALFORMED, both created today (FTB 1)
run status            EXIT 2   107/257 · on the two owned floors only
ledger integrity      CLEAN    107/107 PASS: commit resolvable, impl importable with run(),
                               control_metrics non-empty wherever a control is declared
```

---

## FOR THE BUILDER — ordered, cheapest and most perishable first

1. **Repair the two MALFORMED rows — this is the only thing standing between the
   board and EXIT 0, and it is a two-line edit.** `docs/REVIEW_QUEUE.md`'s
   `ROUTED:` lines for `w0-too-shallow` and
   `goal-187-names-seven-primitives-four-have-no-commitment` carry 5
   pipe-separated fields; the parser allows 4
   (`review_queue.py:371`). **Do NOT delete the superseded disposition** — that
   was the desk's honest instinct and deleting it would be the worse repair.
   Move it out of the pipe-delimited field and into the row's indented body,
   where prose belongs, leaving field 4 as the single current disposition
   (`ACTED 2026-10-17 a24cdb7 (...)` and `ACTED 2026-10-17 2fd7de5 (...)`).
   Verify by calling `review_queue.parse()` and asserting, for both rows,
   `bad == []`, `status == 'ACTED'`, and `routed == 2026-08-24` / `2026-09-16`;
   then `run review-queue` must print `142 routed` with the counts summing to
   142 and **EXIT 0**. Price the `REVIEW_QUEUE.md` edit bill before editing, as
   always, and pay it in-slot.

2. **Transcribe `D43`'s firing onto `docs/DECISIONS_RESOLVED.md`.** It fired this
   audit and `docs/DECISIONS_RESOLVED.md` is outside my write permissions, so
   `decisions --check` prints `31 of 32 ... UNTRANSCRIBED: D43`. It does **not**
   break a ratchet and does **not** fail a gate — which is precisely why it gets
   forgotten; `D36` took nine days this way. The firing record in
   `docs/DECISIONS_NEEDED.md` is complete and self-contained; copy it in the
   idiom of the `D36`/`D37` records at `5bc466f`. Add nothing and rule nothing.

3. **Make the `pass_on_dead_dependency` ratchet note explain the 6 it records.**
   `experiments/ratchet_readings.json`'s note is byte-identical to the one
   stored against value 5 on 2026-09-26, while `value` is 6 and `at` now reads
   2026-10-17. Append one sentence naming the 6th pair — **T1.12 ← T1.11 FAIL,
   arrived 2026-10-04** — and its owner, `t112-pass-stands-on-t111s-orthogonal-
   added-conjunct` (OPEN, DUE 2026-10-18). **Do not raise
   `PASS_ON_DEAD_DEPENDENCY_BASELINE`.** The explanation already exists in your
   own journal and in `b0a3d56`; this moves it to the artifact a future reader
   actually opens.

4. **A note on item 1's generalisation, for whoever writes the repair:** the
   format has no way to carry a superseded disposition, and the desk needed one
   today. If you see a cheap, monotone way to let the row body hold
   `SUPERSEDED:` history in the same idiom as `DUE:` and `BLOCKED-BY:`, say so
   on a queue row rather than building it — `review_queue.py` is a governance
   instrument and the STANDING FREEZE applies.

---

## FOR THE OWNER

**1. `D43` is off your desk by default, not by answer — and the one number it
asked you to decide on was wrong.** The owner did not rule by 2026-10-16, so the
pre-registered default fired: no world-edit window is declared, and the seven
measured findings across `NE.01`, `SH.02`, `HR.5`, `BA.03` and `W.1`/`W.2` stay
DECLINED. Firing it changed nothing — the world was never edited
(`playground.py` untouched since 2026-08-19) and no row was waiting. **Reversal
is one sentence from you at any time**, and the row that would use it is alive
(`seven-rows-are-held-behind-a-refused-window-and-a-moot-decision`, OPEN, DUE
2026-10-25). The correction: the entry prices the window at *21 PASS
certificates*; it measures **35** — 2.88 CPU-hours and 1.52 GPU-hours, four
builder slots. That strengthens the entry's own argument (pay 35 once, not seven
times), but you were asked to weigh the wrong figure. Ruling that the window may
**never** exist is also a real answer and closes the class honestly; leaving it
to silence is what just happened.

**2. NO-DECISION — your Review wrote you a question yesterday and nearly lost
it, and the near-loss is the thing worth your attention.** `D47` (armed, `decide
_by` 2026-10-24) asks whether the desk's docket should be split from the
builder's so the two are measured separately. The Review's reasoning is sound
and I have not touched it. What I want on the record is how it reached you: the
Review died at max turns at 12:42 holding the entry **uncommitted**, and in a
form that would have broken a ratchet if committed as written. It survived
because this audit diffed the working tree. **`D44` — the character-sheet
question, the only real discovery about Jack in a fortnight — sat thirteen days
in a queue row for the same class of reason and is still unanswered, due
2026-10-24.** Two of your organs now routinely die before they can finish
telling you things.

**3. NO-DECISION — the compute pairing, counted this audit, not inherited.**
`2026-W41`: **30.0 free Kaggle GPU-hours, 0.000 drawn, expiring tonight** — the
fifth consecutive week, ~141 of ~150 hours across W37–W41 gone unbought. And the
honesty `D40` insisted on: **those hours had no designed buyer.** All three GPU
cost classes read `EMPTY`/`NOT FILLABLE` mechanically, and the `T1.08` prohibition
forbids the board's only recent buyer. The hours are not the argument. The
argument is that `T1.08` is FAIL, blocks 45 specs, is out of the builder's hands,
and its residual is a desk estimator question dated 2026-10-21 — so the ladder
cannot move and the quota cannot be spent, for the same single reason.

**4. NO-DECISION — six decisions are marked as yours to execute at the desk and
are stale: `D33` (24 days), `D35` (23), `D38` (13), `D40` (7), `D39` (2), `D46`
(today).** `decisions --check` says of each *"execute it, report it, do not
ask"*. I raise them only because `D33` is the upstream whose stop-rule produced
the seven rows `D43` just closed by default — the chain ends on a desk entry
nobody has run for over three weeks. Nothing here is for you to rule on unless
you judge that these are in fact yours, in which case say so and they will be
answered rather than executed.
