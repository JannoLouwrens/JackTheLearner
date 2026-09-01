# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the anatomy audit and the completeness audit are Sunday
> work and were deliberately skipped; the last FULL page is 2026-08-31, and its
> findings below are tracked, not repeated).

**2026-09-01 06:4x–07:2x UTC — DAILY.** Window: the last 24 h
(2026-08-31 06:4x → 2026-09-01 06:4x), which contains yesterday's FULL run at
19:1x as an interior point.

*The one sentence: **the builder executed the entire FULL review in a single
day — every hard item, at full speed, honestly — and the result is that the
ladder now has two units in flight, nothing else to dispatch, and almost
everything that remains sitting on this desk waiting for Sunday.***

---

## The numbers

| | now | 08-31 19:5x (FULL) | Δ |
|---|---|---|---|
| demonstrated / registered | **93 / 213** | 94 / 211 | **−1 / +2** |
| pass rate | **43.7%** | 44.5% | −0.8 pts |
| FAIL / VOID | **18 / 7** | 16 / 5 | +2 / +2 |
| unreachable specs | **81 / 213** | 80 / 211 | +1 |
| rework (ledger rows at attempt > 1) | 77 / 119 = **64.7%** | 66.1% | −1.4 |
| commits, last 24 h | **62** | — | — |
| builder slots fired, last 24 h | **24 / 24**, zero `PACING:` skips | — | — |
| ledger settlements, last 24 h | **14** (5 PASS / 6 FAIL / 3 VOID) | — | — |
| all four ratchets | coverage rc=0 · champions ok · decisions ok (0/10 undeclared) · review-queue 0 violations | same | — |

**The number that matters is not on that table, and it is this: of the five
PASSes in 24 hours, five are re-buys of specs that already passed.** `T0.10`
attempt 3, `W0.DIAG` attempt 3, `T0.29` attempt 4, `T0.17` attempt **22**,
`T0.21` attempt **22**. Every one is a certificate being re-purchased after an
`IMPL_DEPS` decay, which is the machine working as designed and is also, to the
creature, nothing at all.

**Since the FULL run ended at 19:5x — eleven hours — there have been six ledger
settlements and every one is red:** `ME.11.E` VOID-FORECLOSED, `ME.11.F`
VOID-FORECLOSED, `LT.01` FAIL, `T2.10` FAIL, `T0.27` FAIL, and `UB.14` VOID
(landed 06:41:04, while this Review was reading the log). That is not a
complaint. Five of the six were *ordered*, and the sixth was forecast in
writing before it ran.

**Goodhart check: the rate fell, and this time it fell for the best possible
reason.** The single demonstrated capability lost is `T2.10` — *"Memory
retrieval beats recency"*, the founding certificate of the Episodic-retrieval
seat, PASS since 2026-08-08. Yesterday's FULL run found it resting on a venue
the project has since proved false and routed a strictly-harder conjunct to the
builder. The builder implemented the conjunct (`468772e`) and re-bought the
spec (`b4805ac`), and it came back **FAIL**: the original venue is still perfect
on all three seeds (recall@5 1.0 vs null 0.178, latest@1 1.0, simonly 0.0) and
the paraphrase venue reads **0.0000** with the instrument proven alive (leaky
1.0). **A capability left the scoreboard because we asked it a harder question
and it could not answer.** That is the ladder doing the only thing it is for.
The rate fell; the honesty rose; do not read the two as the same sign.

**Rework fell 1.4 points and is still not the problem.** Same reading as the
last three pages: attempts 2+ are VOID→repair→re-run and certificate re-buys
after deliberate tooling edits.

---

## The frontier, recomputed — and for once the builder is standing on it

**Transitive-block mass (`run blocked`, recomputed):** 81 of 213 unreachable.
`T2.01` is still the largest single blocker — **frees 35, blocks 38**.

**And `T2.01` is, for the first time, downstream of something in flight.**
`D1.0` — the four-arm control-path bakeoff that D1's default registered
yesterday, discharging a phantom arena that had been cited by `CHAMPIONS.md`
for 22 days — is **RUNNING on Kaggle**: kernel 1 of 3 harvested `ok` at 14,674 s
charged, kernel 2 dispatched 06:17, watcher process verified alive at 4 h 26 m.
Its winner is the architecture under which `T2.01` and `T2.02` re-run. **The
builder is working on the largest mass on the board.** After eleven days of
this desk saying otherwise, that deserves to be said plainly.

**The rest of the board is empty, and the emptiness is structural.** `coverage`
reports QUEUE DEPTH 3, of which **2 are fresh dispatches and both are in
flight** (`D1.0`, `UB.14` — and `UB.14` settled VOID mid-review, so as of now it
is one). Four cost classes have **no path in at all**: nothing to implement and
nothing to pilot. Five specs are VOID-FORECLOSED, four are PILOT-BLOCKED, five
are gate-provisional. Every one of those repairs is a redesign, and **every
redesign is routed to this desk**: `run review-queue` reports **11 OPEN rows,
four of them DUE 2026-09-06**.

**So the honest statement of the frontier today is uncomfortable: the builder is
no longer the bottleneck. This organ is.**

### The one actionable unblock the builder can still take, and why it was invisible

**`UB.10` — `NOT_RUN`, frees 4, blocks 5 — has been PARKED since 2026-08-20,
and the Review dispositioned its unpark on 2026-08-25.** The
`recipe-sensitivity` row carries a full design (matched *tuning budget* rather
than matched hyperparameters: identical pre-registered LR grid, identical trial
count, identical selection criterion, all declared before any arm runs; an arm
that clears `uni_learn_ok` nowhere is SCORED-AND-INELIGIBLE, not silently a
0.5 — strictly harder, cost N → N×K). Seven days later the spec is still parked
and `coverage` still excludes it.

The cause is one word. That row's status reads **`ACTED 2026-08-25 (design in
docs/PROGRESS.md § FOR THE BUILDER item 2)`** — where `ACTED` means *the Review
acted*. On the `me11-…` row directly below it, `ACTED` means *the builder
executed*. **Same token, two meanings, and in the first sense it closes a row
whose work has not started.** A top-down reader sees a settled row; `run
review-queue` prints `ACTED 12 d` and no violation; the park never lifts; five
specs stay welded — including `UB.11`, which Review 08-31 item 4 needs before
the `T2.12` fusion-boundary conjunct can even be written.

Installed as the builder's next fresh unit. **The generalisable defect —
a two-meaning status token in a machine-read file — is FOR THE BUILDER item 4
below**, because a fix in prose here would rot exactly the way the row did.

---

## Part 2.5 — steering maintenance

**1. `scripts/ladder_prompt.md` PRIORITY block: REPLACED (spent, again, in one
day).** The block installed by yesterday's FULL ordered two things: settle
`ME.11.E`/`ME.11.F` without running them, then run `LT.01`. **Both were done
within eight hours.** A priority block naming completed work is the failure this
duty exists to prevent, and it has now occurred on three consecutive days —
though for the happiest of reasons, which is that the builder is faster than its
map. The new block: (i) names the two in-flight units and forbids starting work
against them; (ii) installs `UB.10` as the next fresh unit with the 08-25
disposition quoted, so the builder does not have to find it; (iii) sequences the
three unhandled Review 08-31 items (6, then 4, then 5); (iv) states, without
ordering any action on it, that the `w0-too-shallow` evidence has changed shape.
No count or status is cached on the page — every claim points at a live reader.

**2. `docs/FIELD_WATCH.md`: unchanged since the last review** (last commit
`469bbaf`, 2026-08-31 05:53, wk5), and wk5 was consumed in full by the 08-31
DAILY — N1 accepted narrowed and split, N2's arm deferred with its
shuffled-partner control accepted as an `NE.07` strengthening, N3 accepted and
made runnable as `W0.DIAG` (which then PASSED). **Nothing to consume. The field
watch fired on schedule yesterday (Monday); next report due 2026-09-07.**

**3. Seat staleness — one finding, and it is one day old.** The
**Curiosity-signal seat** is held **BY ANALYSIS** and has never been defended.
Its arena — `LT.01`–`LT.09` — was registered on 2026-08-31, and `LT.01` FAILED
on 2026-08-31. **The seat's ring opened and welded shut inside a single day**,
and `LT.03`/`LT.04` — where `disagree` and `metra` were finally to race `lp` on
GOAL.md's north star — are behind it. Routed as its own row (below). Everything
else: the **Learning core** seat is contestable today via `LC.07` (registered
yesterday, deps all PASS, gates provisional pending a throughput pilot); the
**Control architecture** seat's arena `D1.0` is running; **Vision encoder** and
**Emotion** remain BY DEFAULT and uncontested, both with real arenas and neither
progressing — carried, not new.

**4. Organ liveness — all four alive, none silent past its cadence.**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 2026-09-01 06:26 | **alive**, 24/24 slots in 24 h, 0 pace skips |
| overseer | 6-hourly | 2026-09-01 06:37 | **alive** (prior 00:45) |
| field watch | Mondays | 2026-08-31 05:53 | **alive**, on schedule |
| review | Sundays + daily | 2026-09-01 06:37 | **alive** — this run |

And the background work is **verified running, not asserted**: `UB.14`'s
recorded run (pid 49101) was alive at 14 minutes and settled VOID at 06:41:04;
`D1.0`'s watcher (pid 4187660) alive at 4 h 26 m with kernel 2 dispatched at
06:17:14. `experiments/gpu_budget.json` and `gpu_submissions.jsonl` are dirty in
the working tree — that is the live watcher writing, correctly not committed by
me.

**5. New review-queue row opened: `lt01-c2-body-cannot-rise`, DUE 2026-09-06.**
The builder attached `LT.01`'s finding to `w0-too-shallow` as an UPDATE, and for
the instrument *count* that was exactly right — assembling the aggregate in one
place was the 08-31 finding and the builder applied it correctly the next day.
But the owed **action** is not a count. `run review-queue` prints row titles,
not row bodies, so an owed redesign living 200 lines inside another row is the
precise shape of `wk4-N3` — ordered as prose on 08-25 and read by nobody for six
days. It now has a title, a `DUE:`, three declarable options, the threshold rule
written on its face (0.6 m may not be lowered), and the circularity computed
below. Ratchet re-verified: **15 routed, 0 violations, rc=0.**

---

## The honest paragraph

Closer, and by a route worth naming precisely, because it is not the route
anyone would have designed. Yesterday this desk wrote that the builder was fast,
honest and pointed at the wrong thing. Overnight it turned to face the right
thing and demolished almost everything in front of it — settled the retrieval
family it had refuted with its own hands, built and ran the curiosity ladder's
entry point, redesigned the certificate this desk said was resting on a false
venue and watched it fall, and discharged eleven decisions that had been armed
and waiting for someone to be brave enough to let their defaults fire. Nothing
about that day was cowardly and nothing about it was slow. And what it produced,
measured in capability, is nothing: the board is emptier than it was, the
scoreboard is shorter by one, and the creature cannot do a single thing today
that it could not do yesterday. That is the correct outcome of a day spent
subtracting, and it is also the clearest sign yet of where the project actually
sits — we can falsify faster than we can build, and we have now falsified our
way to the bottom of the queue. The single most important step toward Jack was
letting `T2.10` fall: a founding certificate died because we finally asked it
the question we had learned to ask, and nobody flinched. The most concerning
drift is that the thing which stopped him moving forward this week was not a
missing capability or a failed experiment but a park, a default, and a word in a
status field that meant two different things — three pieces of bookkeeping, each
individually correct, which together have his body waiting on a spec that cannot
run because of his body. He is not blocked by physics. He is blocked by
paperwork that has never been read end to end by anything that could notice.

---

## REWRITTEN / STRENGTHENED

| spec / file | change | why it is stronger |
|---|---|---|
| `scripts/ladder_prompt.md` | priority head block **replaced**: yesterday's order was fully executed in 8 h, so its central instruction named finished work; the new block pins the two in-flight units, installs `UB.10` with its 08-25 disposition quoted inline, sequences the three unhandled 08-31 items, and states the changed shape of the `w0-too-shallow` evidence as context the builder must not re-measure | the map was again ordering completed work; and the unpark that frees 4 specs was invisible from every top-down read, so quoting it into the builder's own file is the only place it gets seen |
| `docs/REVIEW_QUEUE.md` | **new row `lt01-c2-body-cannot-rise`**, DUE 2026-09-06, with three declarable options, the threshold rule on its face, and the D8/D9 ↔ `LT.08` circularity computed | an owed redesign that only existed as an UPDATE paragraph inside another row is unreachable by the queue's own reader — the exact defect that lost `wk4-N3` six days |
| `docs/PROGRESS_LOG.md` | one row appended | trend line continues |

**No threshold moved. No control softened. No FAILING or VOID spec was
rewritten. No spec file was touched by this run** — Part 2 is Sunday work and
was skipped deliberately.

---

## FOR THE BUILDER — ordered

1. **Land `UB.14`'s `VOID-FORECLOSED` declaration first.** It settled VOID at
   06:41:04 and your own 06:26 commit says the declaration is owed at harvest,
   quoting the row's fired conjuncts and priced with `FORECLOSURE ARITHMETIC`
   and `BLAST RADIUS`. The 54th audit's B3 refuses an unpriced one loudly in
   both readers. Small, owed, and the only thing between an honest measurement
   and a silent one.

2. **Then `UB.10` — un-park it under the 2026-08-25 disposition.** `NOT_RUN`,
   **frees 4, blocks 5**, the largest mass on the board that is neither in
   flight nor on this desk. The design is already written and is strictly
   harder than what it replaces (matched tuning budget: identical LR grid,
   identical trial count, identical pre-registered selection criterion, all
   declared before any arm runs; `uni_learn_ok`-nowhere ⇒ SCORED-AND-INELIGIBLE,
   never a silent 0.5). It has waited seven days because its queue row reads
   `ACTED`. It also unblocks `UB.11`, without which item 4 below cannot be
   written.

3. **Do not disturb `D1.0`.** Kernel 2 of 3 in flight, watcher alive, envelope
   frozen from a harvested pilot with a pre-registered branch tree. It is the
   single largest unblock in the project (`T2.01`, frees 35, re-runs under its
   winner). No second GPU job against it, no envelope re-derivation.

4. **Fix the two-meaning `ACTED` token in `docs/REVIEW_QUEUE.md` — with teeth,
   not prose.** On `recipe-sensitivity`, `ACTED` means *the Review produced a
   design*; on `me11-…`, it means *the builder executed one*. The first sense
   silently closed a row whose work had not started and parked a frees-4 spec
   for a week. The repair is a distinguished status —
   `DISPOSITIONED` (design exists, execution owed) vs `ACTED` (executed, commit
   named) — taught to `experiments/review_queue.py` so that a `DISPOSITIONED`
   row **keeps ageing and can go OVERDUE**, with a known-answer fixture and a
   sabotage check, per house style. `T0.31` gates the reader, so it will need a
   re-buy. This is the only structural item on the list and it is the one that
   makes the class of bug unrepeatable.

5. **Register `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`** (Review 08-31 item 6,
   untaken). GOAL.md cites four spec ids that do not exist; `coverage` has
   reported them DANGLING since 2026-08-25 and 273 commits have passed. One
   iteration, clears a dangle, and until it is done the constitution cites
   vapour.

6. **`T2.12`'s fusion-boundary conjunct** (Review 08-31 item 4) — **after
   `UB.10`**, which is its precondition. Unchanged in substance: keep both
   existing controls, add that PAD separability must survive at the fusion
   boundary in a live `UB.11` ablation. Strictly harder; it converts
   "separable from noise" into "load-bearing", which is GOAL.md's own standard
   for every sense.

7. **`T1.09`/`T1.10` re-aimed at the P100** (Review 08-31 item 5) — **after
   `D1.0` clears the GPU queue.** Both still name a T4 this project has not run
   on since 08-12. Same ceilings, correct device; not a weakening.

---

## FOR THE OWNER — one fork, and it is new as of yesterday

### The body is parked behind a spec that failed because of the body

Two armed defaults fired yesterday, each correct on its own terms, and together
they closed a loop that neither of them can see.

- **`D9` fired: PARK the rover-body question** until *"the playground-humanoid
  line."* It adopts nothing and re-runs nothing — the cleanest available branch,
  and I do not dispute it.
- **`D8` fired: re-parent `BA.02`** behind **`LT.08`**, the registered spec on
  which a body with directional catch authority arrives. Also clean.

**`LT.08` is the playground-humanoid line, and `LT.08` is unreachable.** Its
`depends_on` is `[LT.07, T2.01, T2.02]`. `LT.07` sits at the end of the chain
`LT.01 → LT.03 → LT.05 → LT.07`. And **`LT.01` failed yesterday on a clause
about the body**: C2 pre-registered that a random agent reaches ≥ 0.6 m of
non-ladder torso rise — the 2026-08-09 pilot's free-roam ceiling was 1.007 m —
and the as-built body recorded `nonladder_rise_max` **0.084 ± 0.067 m**, because
it tips over within seconds and travels by dragging. Every aliveness guard was
green and the other three clauses all held, so the instrument is certified and
the body is what failed.

**So: the body question is parked until a spec chain completes whose first link
failed because of the body.** `T2.01` (frees 35) sits in the same `depends_on`
and has been a FAIL since 08-14, reclassified by this desk from a compute
problem to a science problem. Neither default created this on its own; it is a
joint property of two decisions, a dependency edge and a failed clause, and
nothing in this repository reads those four things together.

**This is the third and fourth measurement of the same thing in eight days.**
`W0.BAL` (08-21): arm C upright **1.000** against the as-built body's
**0.002–0.004**. `BA.03` (08-31): the blind twin holds **98.9%** of the horizon,
so the balance sense is not merely unhelpful but unnecessary. `LT.01` (08-31):
the body cannot gain a tenth of the height a random agent was expected to reach.
`UB.14` (today, VOID): the eye reads its own body's position at **0.159**
held-out against a 0.5 gate — a geometric ceiling inside the ±0.4 m box a 30°
half-FOV allows. **Three of the eleven `w0-too-shallow` instruments are now
measurements about the body rather than the world, and two of the three landed
in the last twenty-four hours.** The row is named for a hypothesis about the
world, and a world redesign does not repair a body that tips over in seconds.

**My recommendation, unchanged in substance from yesterday's FOR THE OWNER 1 and
now considerably more urgent: register `W0.BAL` as a spec id and seat the body.**
I could not do it myself — `W0.BAL` is a bakeoff, not a registered id, so the
chair would arrive `NO-ARENA` and push a ratchet the wrong way. **I am not asking
you to un-park `D9`'s adoption.** I am asking that the parking be *visible as a
parked seat* rather than as silence, because right now a body we have measured
cannot stand is the unnamed incumbent of a seat that does not exist, and the
question of whether it can act is scheduled to be asked by a spec it has already
made unreachable.

If you would rather not create the seat, the alternative that costs least is to
tell me at 09-06 which of the three options on `lt01-c2-body-cannot-rise` you
want the design to take — because the circularity is only broken from one of
three places, and two of them are yours, not mine.

### Carried, unchanged, from 2026-08-31 (not re-argued here)

- **Re-tier `T6.03` out of Tier 6.** A save/load round-trip bought on day four
  should not be a green tick in the tier GOAL.md calls the finish line.
- **The builder's budget is the untested twin of the Review's.** Four of the
  seven max-turns deaths across all organs are the builder's, and it is the only
  organ whose budget was never derived from its scope. Not a decision request.
- **The next real FULL run is 2026-09-06.** Cron still keys the Sunday branch off
  `date +%u = 7`; yesterday's run proves the raised budget *fits*, not that it
  *fires*. Four rows are due that day, and one of them is the world design that
  has been re-armed once already. **Week 3's rule binds me: a third deferral
  would be a lie.**
