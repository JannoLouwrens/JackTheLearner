# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-29 00:37–01:0x UTC — the 129th audit.** Six hours after the 128th
(18:37–18:5x). The window is the builder's slots `19:07` through `00:07` and
demonstrated moved **107 → 107**.

Instrument exit codes, re-derived this sitting from a clean tree and not
inherited: `coverage` **2**, `decisions --check` **1**, `champions --check`
**0**, `run status` **2**, `run review-queue` **2**.

Ratchet delta, quoted as a DELTA before anything is recorded, from `run status`'s
own SLOT LINE: **5 MOVED** (`fail_unowned_owned_forms` queue-row 29 → 30,
`review_queue_net_arrivals` 26 → 30, `review_queue_violation_forms`
`{OVERDUE:1}` → `{HOLD-ON-A-RESOLVED-BLOCKER:9, OVERDUE:6}`,
`review_queue_violations` 1 → 15, `unreachable` 95 → 96); 1 day-rolled
(`cpu_foreclosed_now`, the clock); no counter refused to compute; floors
**3 ABOVE** (`decisions_default_action_expired`, `pass_on_dead_dependency`,
`unreachable`), 0 BELOW, 0 UNVERIFIED.

**The OVERDUE +5 is the calendar, and it was pre-registered to the row.** The
22:0x slot named the six rows that would flip at midnight (`ub10-part1`, `lg12`,
`so10`, `lg13`, `lc03-five-controls`, `d27-screen`) and the 00:1x slot verified
the flip exact: **15 = 6 + 9**. That is a forecast made in the open and
confirmed against the instrument, not a number that moved while nobody looked.

**DISCLOSURE — the clerical cron fired at 00:43, six minutes INTO this
sitting**, and committed `a92dd14` while I was reading. It **ADMITTED** after
the midnight day-meter reset, exactly as the 21:0x slot pre-registered and the
22:0x slot verified at source: re-bought `T0.21` (PASS 9.98 s, hash-salt
differential) and `T0.28` (PASS 46.76 s), *"no status moved"*, and printed its
two refusals (`T6.03`, `T0.18` — *"stale but unpayable"*). Both rows carry
`dirty_files = None`. The prediction chain was correct end to end and I am
recording it as a working forecast, which this project has not had many of.

**What I did NOT do, named rather than omitted.** I re-ran no spec (the
no-dry-run rule). I armed no decision: `decisions_undeclared` reads **0** —
there is nothing armable on the page — and manufacturing an entry to satisfy
the per-audit arming quota is the disease the quota exists to prevent. I
appended nothing to `DECISIONS_NEEDED.md`: nothing this sitting found needs the
owner that is not already routed. Said plainly so the absences are not read as
oversights.

---

## VERDICT: DRIFTING — six slots, six `rc=0`, zero verdict changes, and not one spec about Jack ran. The ledger's arithmetic is clean and I found no loosening. What I did find is that the loop lost a whole slot to a wait no slot can perform, and it set the same trap again four hours later; and that one of the three counters standing above its floor has had no written cause for two days while its two neighbours both have one. Neither damages a certificate. Both are the kind of hole that stays invisible because every instrument reports `rc=0`.

---

## RANK 1 — the `20:07` slot took a unit whose completion condition lay **outside its own lifetime**, ended `rc=0` at 20:11 having committed nothing and written no journal entry, and at 00:16 the builder armed the identical trap again. No instrument counts this: `dark_slots` reads 0 and is correct by its own definition. (HIGH, new this window)

### The measurement

From `/data/jack-logs/ladder.log`:

    2026-09-28T20:07:11  iteration start — 107/254 demonstrated
    2026-09-28T20:11:52  iteration end rc=0 — 107 -> 107 demonstrated

The slot's own summary, verbatim:

> *"Waiting on the 20:43 regate tick now — the background watcher (task
> `bhujagtik`) will wake me when it completes, at which point I'll verify the
> slack guard's behavior, write the journal line, and commit."*

It did not. Three independent checks, each run rather than inferred:

- **`git log` between 19:14 and 21:10 is EMPTY.** The window's commits go
  `e240a50` (19:13) → `667499a` (21:11). The slot produced no commit of any
  kind.
- **`docs/LOOP_JOURNAL.md` has no `20:0x` entry.** The headings run
  `## 2026-09-28 19:0x UTC` (line 23945) → `## 2026-09-28 21:0x` (line 24023).
- **The successor names the gap in its own heading** — *"the one receipt the
  20:0x slot could not have seen — it ended at 20:11, the tick fired 20:43."*

### Why it is structural and not bad luck

The regate cron is `43 */2`. The loop is `7 * * * *`, and its six slots in my
window ran **4 m 41 s to 9 m 45 s**, mean ≈ 7 minutes. A slot that starts at
`:07` and ends by `:17` can never observe a `:43` event: the wait is **26–36
minutes on a process with a ~7-minute life.** No slot can keep this promise,
ever. The plan was not optimistic — it was arithmetically impossible at the
moment it was written.

**And it was written twice.** The `00:07` slot ended at 00:16:56 on:

> *"Now holding for the 00:43 regate tick to consume the predecessor's ADMIT
> prediction; the watcher will wake me when it lands."*

The tick fired 26 minutes after that slot was already dead. That slot lost only
its unit — it had committed `5651fc1` + `3a45cb9` first, which is why it does
not read as empty. The `20:07` slot lost everything it did.

### Why no organ can see it

`dark_slots` (`scripts/usage_attribution.py:222-261`) counts **trailing
`PACING:`/`STOPPED at` lines** — slots the loop declined to run. A slot that
starts, runs, ends `rc=0` and commits nothing is not dark by that definition,
and the counter is right to read 0. `lost_iterations.log` is 0 bytes and also
correct: nothing was lost to a limit. `run status` sees no change because
nothing changed. **So the project's three liveness readings all report health
about a slot that produced no artifact**, and the loss is legible only by
diffing journal headings against `ladder.log` — which is precisely the class
`docs/LESSONS.md` already records as *journal gaps hide the losses*.

### And the failure mode is that it reads as diligence

The `21:0x` slot opens: *"this iteration's one unit of work was the verification
its predecessor could not perform."* That sentence is true, generous and
exactly wrong as an accounting: the predecessor did not lack the *information*,
it lacked the *lifetime*, and describing the inheritance as care removes the
reason to stop doing it. One slot in six — **17 % of my window** — produced
nothing, and the record reads as six good slots in a row.

**I am not criticising the verification itself.** The 20:43 guard check was
worth doing and the `21:0x` slot did it well. The defect is the *scheduling
decision*, and it is one line of conduct to fix.

---

## RANK 2 — `unreachable = 96` has stood above its floor of 95 for two days with **no cause written at its site**, while both of its neighbours above their floors carry one. The cause is one step from `run blocked`: it is `LT.03`, bought by `LT.02`'s honest demotion. Three audits have called it "inherited" without naming it. (MEDIUM, new this window — the number is not, the attribution is)

`experiments/ratchet_readings.json` holds, verbatim:

    "unreachable":                        {"at": "2026-09-25", "value": 95}
    "pass_on_dead_dependency":            {"at": "2026-09-26", "value": 5,
                                           "note": "3 -> 5 on 2026-09-26 ..."}
    "decisions_default_action_expired":   {"at": "2026-09-26", "value": 1,
                                           "note": "`at` dates the KEY ..."}

Two of the three above-floor counters explain themselves to every reader of
`run status`. The third prints `!! ABOVE its declared floor 95 — growth nobody
raised the constant for` and stops. My own two predecessors and the 126th each
recorded it as inherited and re-derived; none of them said **which spec**.

**It is `LT.03`, and the derivation is two readings:**

- `run blocked`: `LT.02 = FAIL  frees 1  → LT.03`. "Frees 1" means `LT.02` is
  `LT.03`'s *only* unsatisfied blocker — confirmed against the registry,
  `LT.03.depends_on = ['LT.01', 'LT.02', 'PG.4']`, and `LT.01`/`PG.4` both PASS.
- `run status`'s own settle-events: `2026-09-25T13:25 LT.02 PASS CHANGED`, then
  `2026-09-27T02:40 LT.02 FAIL CHANGED`. The recorded reading of 95 was taken
  **2026-09-25**, while `LT.02` still passed and `LT.03` was still reachable.

So `unreachable` went 95 → 96 at 2026-09-27T02:40, and the +1 is `LT.03`
becoming unreachable behind an **honest** FAIL — the same shape as
`pass_on_dead_dependency` 3 → 5 behind `T0.13`, which got a note. The
arithmetic is confirmed running the other way inside my own 7-day diff:
`88762a2`'s blast-radius line reads *"unreachable 96 -> 95 ... REGAINED
LT.03"*.

**And the demotion itself is sound — the builder proved that at 22:0x** and I
did not take it on trust: `f047060` forecast the re-buy would stay PASS, the
run recorded FAIL 24 minutes later, and `03af51f` found the recorded 5.2631
ratio was `5.2631e-9 / 1e-9`, an epsilon fabrication. The red is real.

**The repair is the note, and explicitly NOT the floor.** `UNREACHABLE_BASELINE`
stays 95. `coverage.py:1186` already carries the rule in the steering page's own
words — *"Do not raise UNREACHABLE_BASELINE to cover your own work"* — and this
growth is not the builder's work to cover; it is an honest demotion's
downstream, which is exactly what a note is for.

---

## RANK 3 — a hypothesis I checked and am **refuting rather than reporting**: the overseer's cron and the unattended ledger writer are scheduled six minutes apart on every single sitting, and the dirty-stamp hazard that implies is already closed. (NO FINDING — recorded because the near-miss is on the 128th's page and someone will re-derive it)

`crontab -l`: `37 */6` (overseer, so 00:37 / 06:37 / 12:37 / 18:37 — all even
hours) against `43 */2` (regate, every even hour at `:43`). **The collision is
100 %, by construction, on every audit this organ will ever run**, and an audit
takes 20–40 minutes. The 128th audit disclosed the same overlap as a lucky miss
at 18:43. I watched it fire: at 00:43:24 the tree carried `M
experiments/ledger.json` and `M experiments/cpu_budget.json` — the lane
mid-write while I was reading.

It is not a hazard, and I checked instead of assuming. `docs/OVERSIGHT.md` is in
`protocol.PROSE_DOCS`, therefore in `NOT_CODE`, therefore invisible to the
`+dirty` stamp — corrected into that list on 2026-09-26 by the 121st audit's
FINDING 3, in the same repair that made doc dirt **per-spec** (`b4df9bb`). An
overseer draft sitting in the tree at `:43` cannot stamp anything. Verified at
HEAD, not from the commit message.

**One narrow residue, stated so it is not rediscovered as a crisis.** The one
file this organ may write that is *not* prose-exempt is
`docs/DECISIONS_NEEDED.md` (`INSTRUMENT_INPUT_DOCS`), and `T0.28` declares it —
so an audit appending to the register between `:43` and the sweep's commit
would dirty-stamp `T0.28`'s row, per-spec and nothing else. The window is
~90 seconds, twelve times a day. I did not append this sitting. Not worth a
repair; worth knowing before someone appends at `:43`.

---

## RANK 4 — the window's work: 6 slots, 6 `rc=0`, one spec edit (strengthen-only), all three of the 128th's FTB items discharged and verified in source, and **zero verdict changes**. The builder is not the constraint and has not been for a week. (MEDIUM — drift, and the builder is not its cause)

**All three FTB items verified on disk, not from the journal:**

| item | discharged | how I checked it |
|---|---|---|
| **1** — correct the `T0.28` cause, state the expiry | `e240a50` | `REVIEW_QUEUE.md:11900-11931`. The correction is complete and names the right mechanism. **And the builder made the harder call correctly**: it wrote the receipt in the **prose idiom** with the reason on its face — *"deliberately NOT the declared `BUILDER-TRACE:` field: that field asserts DELIVERED — AWAITING STAMP, and nothing this row owes has been executed."* `DELIVERED` reads 5, unchanged. A declared field there would have been a false claim, and it declined to make one |
| **2** — budget check + measured cheapness in `regate.py` | `e240a50` | guard (a) `crossing_a_class_slack()` at `regate.py:164-181`, reading `class_slack()`'s tightest row, with every declined spec printed at `:203-211`; guard (b) at `:145-159`, recorded runtime vs `CHEAP_NOMINAL_S`. **Both verified live on cron rather than at ship time**: three consecutive DECLINEs (20:43, 22:43 and the 18:43 tick) printing *"projected 51 s on 4,555 s spent > `cpu<2h` slack 3,600 s"*, then a correct ADMIT at 00:43 after the reset. `LT.02` (838 s against a 600 s label) is now caught |
| **3** — strike or substantiate `regate.py:3` | `e240a50` | struck, and struck honestly: the docstring now reads *"the BUILDER's own measurement, 2026-09-28 — no owner act is cited or claimed"*, and names `D19` as the only owner ruling this project has |

**The one spec edit, and it is a strengthening.** `5651fc1`: `T0.31`
`N_PROPERTIES` 22 → 24, adding `p23_a_receipt_is_visible_and_buys_nothing` and
`p24_a_declined_window_is_abandoned_not_opened`, with both names added to the
set the **control must fail**. It ratchets `c4df5a4`'s receipt channel, which
until now was printed and never asserted — the builder's own note says the
ship-time synthetic pins *"were run by hand and never committed, which is
exactly the printed-but-never-asserted rot this spec exists to forbid."* That is
the `diff printed classes against ratcheted ones` lesson executed against
itself. The re-buy is clean: attempt 22, PASS at `2026-09-29T00:15:06`,
`properties_checked = 24.0`, `properties_failed = 0.0`, `dirty_files = None`,
stamped at `5651fc1` — the commit that made the change.

**What the six slots produced, and which `GOAL.md` sentence each serves:**

- `19:07` — the three FTB items above. Serves the first principle's fourth
  clause (*protects the honesty of watching what happens*). Instrument work.
- `20:07` — **nothing.** RANK 1.
- `21:07` — verified the guard's first unattended cron tick. Instrument work.
- `22:07` — verified the `LT.02` demotion end to end and the midnight-reset
  premise at source. `LT.02` is the curiosity family's chaos detector, so this
  is one hop from *he explores because he wants to*; it moved no number.
- `23:07` — **the best-aimed work in the window.** Field watch week 9 §6
  independently reconstructed from the spec's own source rather than the
  draft's script: null floor −1.99 against the formula's −2.000, positive
  controls +0.95/+0.82/+0.40 at the identical 513-param/768-row regime, the
  ρ=0.99 correlated null unmoved at −2.07, the incumbent bar −2.3939 at or past
  the floor and the winner's margin +0.0187 = 6.9 % of the anchor's own 0.2699
  seed spread with one seed regressing. Every number replicates. The conclusion
  — **at most two of five senses are linearly recoverable from the fused
  representation** — is a direct measurement against *one brain, all senses in
  unison*, and it is the only thing in six hours that is about Jack.
- `00:07` — the `T0.31` ratchet above, plus the midnight-flip verification.

**And §6 was deliberately left UNROUTED**, because consumption is the Review's
deferred act and the page it lives on is sealed as an `rc=124` draft. I have
re-read that decision and it is right: routing another organ's unverified draft
under this desk's authority is the thing the seal exists to prevent. What the
`23:0x` slot did instead — verify it so the Review can consume it on solid
ground — is the correct unit. **Recorded here so the verification is not lost
if the draft is rewritten before the Review sits.**

**Creature gate: NONE**, recorded as the violation it is, in every slot. Chain
re-derived each time: `T6.01 ← T4.05 ← T4.04 ← T2.01 ← T1.08 (FAIL)`.
`T1.08`'s repair design is the Review's, `DUE 2026-10-02`.

---

## The mandated sections, with the findings above not repeated

**1. Integrity of the ledger — CLEAN, re-derived not inherited.** All **107**
standing PASS rows carry a `commit` field (**0 empty**); the **57** distinct
commits behind them all still exist in git (`git cat-file -e` over the distinct
set, **0 missing**). Standing classes reported honestly by `run status`:
**2 DIRTY STAMPS** (`T6.03`, `PL.02` — unchanged, both refused with written
reasons), **16 STALE CLAIMS** + 1 pre-`impl_sha` (`T2.02`), **5 UNBACKED
CERTIFICATES** (legal, reporting-only), **1 DELIBERATELY-RED GATE** (`T0.27`,
`live_violations = 3`). **`T0.28` was re-affirmed unattended for the third time
tonight**, still on `live_armed = 1.0` from the single armed entry `D37`; that
is the 128th's RANK 1 now running as steady state, and the forecast of its
2026-10-04 fall is on the queue row, in the journal and in FOR THE OWNER below.

**2. Thresholds and controls over seven days — NO SILENT LOOSENING FOUND.**
In my own window this is nearly trivial and I verified it rather than assumed
it: `git diff f1e6201..HEAD` touches **6 files**, of which exactly one is under
`experiments/tests/` — `T0.31`, and it is the strengthening above. `registry.py`
and `registry_expansion.py` are untouched. Over the full seven days I re-ran the
loosening grep (removed lines matching `_MIN`/`_MAX`/`_FLOOR`/`_CEIL`/seed
counts/comparison operators/` or `) across every diff to those three paths and
resolved every hit to a restructuring whose constants survive in source.
**I deliberately spot-checked a DIFFERENT spec from my predecessor** so the two
audits are not one check counted twice: `T3.06`, whose `RANDOM_DWELL_MAX` was
deleted in `875caf6`. It reappears at `t3_06_ablate_curiosity.py:745` as
**0.0185**, *tighter* than the 0.02 it replaced, derived by
`_derive_random_dwell_cap()` at n = 144, and — the part that matters —
`_assert_dwell_cap_current()` at `:1219-1245` **refuses to run the spec** if the
typed constant and the derivation disagree by more than 1e-9. A threshold that
cannot silently drift from its own justification. **No findings in section 2.**
Stated plainly because it is true.

**3. Drift from the goal.** Covered in RANK 4. The converse, from `coverage`
EXIT 2: **0 commitments with NO declared spec** (floor held), **3 CLAIM-DEAD**
(smell, shelter/building, thermal-kills) and **14 more with live claim specs and
nothing passing** — **17 of the owner's constitutional commitments with zero
passing claims**, including *too cold kills him*, *he builds a shelter*, touch,
tool use, proprioception, sleep, plasticity, fast/slow and the told world.
`NO-LIVE-PATH` stands at **6** distinct commitments/seats; the repair for every
one is a **registration**. `goal_unrunnable = 7`, unchanged since 09-05, with
`GEN.02`/`GEN.03`/`GEN.06`/`GEN.09` flagged **4 NEW unrunnable citations** —
owned by the OPEN row `gen-four-reparented-to-a-decision-that-had-already-closed`
(DUE 10-01). `GOAL.md` cites 16 spec ids, **0 dangling**.

**4. Builder liveness.** 6 iteration starts (`19:07`–`00:07`), **6 ended
`rc=0`**, 0 session-limit deaths, `lost_iterations.log` correctly 0 bytes. PASS
delta **107 → 107** — no verdict changed in six hours. `run next` read **0 fresh
of 51** on every slot, re-derived each time rather than inherited: the sixth
through ninth consecutive verified-empty board. **Nothing was manufactured**, and
I want that recorded as correct conduct for the seventh audit running. The one
defect in the window's liveness is RANK 1, and it is a scheduling choice, not
idleness.

**5. Compute honesty.** Re-derived from `gpu_budget.json`'s own
`charged_jobs` rather than a summary: **`2026-W39` has ZERO charged jobs and
0.00 h spent, against 30.0 free Kaggle GPU-hours expiring Saturday
2026-10-03** — after `W37` 5.25 h and `W38` 0.92 h, this is the **third
consecutive week substantially lost**. Every cost class in `coverage` reads
`NOT FILLABLE`; **3 classes are NEWLY EMPTY** (`cpu<1min`, `gpu<20min`,
`gpu<8h`) with no path in. Both live routes run through `T1.08` (FAIL, blocks
45), whose repair design is undesigned until 10-02. The builder re-derived the
"no legal buyer" claim adversarially at 16:14 and refused the only candidate
with the reason written. **No dispatch has been manufactured and none should
be.** Standing waste unchanged: `gpu_hours_no_verdict` TOTAL **48.42 h**, of
which `D1.0` alone holds **33.78 h across 2 attempts and 0 verdicts**;
`gpu_unattributed_jobs = 21`, AT floor. CPU: the day meter reads **20.5 s at
00:45** against yesterday's 4,554.7 s — the 128th's RANK 2 is repaired and the
repair is working.

**6. Stuck decisions.** `decisions --check` EXIT 1. **No `MEANS-ESCALATED`** —
nothing a measurement could settle sits on the owner's desk; the `D1` disease is
absent. **`decisions_undeclared` = 0, at floor** — nothing was armable this
audit, and I armed nothing rather than invent an entry. One armed: `D37`, due
10-04, default (iii) HOLD, `costs 0 specs` and the entry says so itself. Five
not armed: `D33` (stale 6 d), `D35` (stale 5 d), `D38` — all `CONDUCT-DESK`;
`D33` additionally `DEFAULT-ACTION-EXPIRED`; `D37` flagged `CONDUCT-MISFILED?`.
`decisions_unrouted_owner_ask` and `decisions_vanished_owner_ask` both **0**, at
floor — the `D15` disease this organ measured on 08-29 stays closed, and
`PROGRESS.md`'s single owner-ask is correctly attributed to `D22`/`D33`.

**7. Bakeoff hygiene — `champions --check` EXIT 0, 10 violations, all standing
and none new.** **Learning core** held `BY VERDICT` off `LC.03` (a VOID) with
`VERDICT-IS-A-VOID` **and** `TRIGGER-UNREACHABLE` — that is section 7's *"a VOID
treated as a verdict"*, printed rather than hidden, and `D37` is the live entry
on it. Also standing: **World** `VERDICT-UNDECLARED`/`TRIGGER-UNDECLARED`;
**Fast/slow coupling** `ARENA-UNREACHABLE`/`TRIGGER-UNREACHABLE`; 2 `NO-ARENA`
(ASR, Speaker ID); 2 `UNCONTESTED` (Vision encoder, PLASTIC ONLY).
`champions_unwinnable = 4` AT floor, `champions_trigger_debt = 3`.
**`ARENA-MISSING` remains 0** — my own standing prompt still says 8, and those
seats were repaired by REGISTERING specs, never by deleting arena references,
which is the repair the ratchet was written to force. `KINDLESS DISCHARGE` 1/1
(`LF.02`) unchanged.

**8. THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the
ladder than we were six hours ago?**

No. And this window is the cleanest statement of the problem I have had to
write, because there is nothing to criticise in the work itself.

Six slots. Six `rc=0`. Every mandated read done in order, every board
re-derived instead of inherited, every FTB item from two desks discharged and
verified in source, one instrument strengthened against its own printed-but-
unratcheted rot, three refusals to manufacture work, one refusal to route
another organ's unverified draft, and a pre-registered prediction about an
unattended cron that came true to the spec. That is a machine in good health
doing careful work. **And the demonstrated count did not move, no spec about
Jack ran, and the single piece of work that touched the creature was a
verification of somebody else's finding that was deliberately not routed.**

The one thing that went wrong went wrong in the same grain: a slot chose to
*wait* for a machine instead of *doing* something, and a slot cannot wait.
That is what an organ does when its board is empty and it will not invent work
— it reaches for the only thing left, which is watching. Nine consecutive
verified-empty boards is not a builder problem. It is the measurement that the
work has moved somewhere the builder cannot reach.

Set against the ladder: seventeen of the owner's constitutional commitments
still have no passing claim. Nine live rows sit behind a world whose authorship
was formally DECLINED two days ago and has no owner. Thirty free GPU-hours
expire on Saturday for the third consecutive week, because the only thing worth
spending them on is a design. `T1.08` — one red Tier-1 spec — blocks forty-five
others including every creature gate and every Tier-5 claim, and its repair is
due from a desk on 10-02. Of 31 resolved decisions, 29 were resolved by armed
default and **one, ever, by the owner**.

**We are not closer to Jack. The builder has now spent a full day and a night
proving, carefully and honestly, that there is nothing on its board — and the
thing that would put something there is a design nobody has written and a world
nobody has agreed to author.**

---

## FOR THE BUILDER

1. **Never take a unit whose completion condition lies outside your slot's
   lifetime, and journal the slot that ends with nothing.** This is RANK 1 and
   it is conduct, not code — no new organ, no `D35` question. Your slots run
   ~7 minutes from a `:07` start; the regate cron fires at `:43`. The `20:07`
   slot promised to be woken by a watcher 32 minutes after it would be dead, and
   `git log` for 19:14–21:10 is empty, and `LOOP_JOURNAL.md` goes `19:0x` →
   `21:0x`. You armed the same wait again at 00:16. Two changes:
   (a) **when the unit is "observe a scheduled external event", the correct act
   is to pre-register the check for the next slot and end** — which you already
   do well, and did at 21:0x and 22:0x; just stop adding the hold on top of it.
   (b) **write a `LOOP_JOURNAL.md` entry even when the slot produced nothing**,
   naming what it read and why it ended empty. A slot with no artifact is
   currently visible only by diffing journal headings against `ladder.log`;
   `dark_slots` cannot see it and is correct not to. Do **not** build a counter
   for this — one journal line is the repair, and the queue is running 6.00
   arrivals per cycle against 1.71 disposals.

2. **Write the cause of `unreachable = 96` at its site, and do NOT raise the
   baseline.** RANK 2. `ratchet_readings.json`'s `unreachable` key carries
   `{"at": "2026-09-25", "value": 95}` and no `note`, while both of its
   above-floor neighbours carry one. The cause, which you may quote and should
   re-derive rather than inherit: **`LT.03`**, whose only unsatisfied blocker is
   `LT.02` (`run blocked`: *"LT.02 = FAIL frees 1 → LT.03"*;
   `LT.03.depends_on = [LT.01, LT.02, PG.4]`, the other two PASS), and `LT.02`
   went `PASS → FAIL` at `2026-09-27T02:40` — *after* the 09-25 reading was
   taken. Your own `88762a2` blast-radius line states the same arithmetic in
   reverse (*"unreachable 96 -> 95 ... REGAINED LT.03"*). `UNREACHABLE_BASELINE`
   **stays 95**: this is an honest demotion's downstream, not work to cover, and
   `coverage.py:1186` already says so in the steering page's words. Same idiom
   as the `pass_on_dead_dependency` note that is already there.

3. **Nothing else.** Items 1 and 2 add no queue rows and touch no bar. Your
   `regate` repairs from last night are verified working on cron, including the
   ADMIT after the reset — that lane is closed as far as I am concerned.

## FOR THE OWNER

**1. `D33` — who authors the W1 world edit? Six days past its deadline, and
neither organ can clear the red.** Unchanged from the 126th, 127th and 128th
audits, and I am repeating it because nine live rows are behind it and it is now
the largest single fact about this project. `decisions_default_action_expired`
reads **1** against floor **0**, unchanged since 2026-09-23. `D33`'s own default
is *"re-date once more, to 2026-09-23"*, which cannot fire before 09-24 — so the
act it names is in the past on the day it fires. The Review has formally
**DECLINED** the W1 authorship (the first `DECLINED` in 113 routed rows) and
correctly refuses to move `decide_by`. `D13` bars me from editing the register's
rulings. The instrument names two legal repairs — **shorten `decide_by`** (a
deadline may tighten, never lengthen) or **declare whose date it is with
`(CLOCK: <whose>)`** — and **neither is available to either organ.** Nine live
rows now read `HOLD-ON-A-RESOLVED-BLOCKER`, two of them (`ne01-occlusion-
knife-edge`, `water-apply-phantom-force`) **36 days old with no `DUE:` at all**.
The Review recommends option (ii) — the builder drafts under Review — and says
it may not carve that exception out of `D22`, which is your ruling. A stop-rule
fires 2026-10-09.

**2. PERISHABLE — your headline number is still scheduled to fall by one on
2026-10-04, and it is now being re-affirmed by a machine every two hours.**
`T0.28` requires at least one *armed* decision to exist in
`DECISIONS_NEEDED.md`. It failed on 09-26 when the register legitimately emptied
and passed on 09-28 at `live_armed = 1.0` because `D37` is open. Since then the
clerical cron has re-bought and re-affirmed it **three more times unattended** —
most recently at 00:44 tonight — on that same one-row basis. `D37`'s `decide_by`
is **2026-10-04**; the queue row that repairs the property is OPEN and due the
same day. **Ruling `D37` will subtract one from the demonstrated count**, within
about two hours, by cron. That is an artefact of the certificate's shape, not a
loss. Do not let it discourage the ruling. Nothing here needs your decision —
the repair is a Review disposition and the builder has written the full forecast
onto the row.

**3. `D35`'s three one-line repairs, unchanged from the 116th, 125th, 126th,
127th and 128th audits.** (a) a reachable release condition — `T6.01` sits behind
`T4.05 ← T4.04 ← T2.01 ← T1.08` (FAIL), so the freeze cannot end by any act
available to anyone; (b) a clause-2 exemption for truthfulness repairs and
floors on EXISTING checkers, which would unblock the unfloored
`no_control_specs`; (c) confirm the freeze is meant to be unbounded; (d) the
128th's addition — say whether clause 2's *"audit organ, checker or ratchet"*
covers autonomous **actors**, since it stopped a one-expression read-only join
and did not reach a cron lane that writes the ledger.

**4. NO-DECISION, standing report: `2026-W39` has ZERO charged GPU jobs and 30.0
free Kaggle hours expiring Saturday 2026-10-03 — the third consecutive week, and
no dispatch should be manufactured.** Re-derived from `gpu_budget.json`'s
per-job records (`W37` 5.25 h, `W38` 0.92 h, `W39` no jobs at all). Every cost
class reads `NOT FILLABLE` and three are newly empty with no path in. Both live
routes run through `T1.08` (FAIL, blocks 45), whose repair design is the
Review's and is due 10-02. Standing waste unchanged: `D1.0` holds **33.78
GPU-hours across 2 attempts and 0 verdicts**. The scarce resource is a designed
unblock and your ruling, not a machine hour.

**5. NO-DECISION, liveness: 0 dark slots, 6 of 6 builder slots `rc=0`, and one
slot that produced nothing without any counter noticing.** All four organs fired
within cadence. The builder has now verified an empty board nine consecutive
times without inventing work, which is the correct behaviour and is also the
measurement that the bottleneck is elsewhere — it is the two design debts in
item 1 and item 4, both of which sit at a desk that sits for twenty minutes a
day.
