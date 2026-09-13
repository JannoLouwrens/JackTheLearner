# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **93rd audit — 2026-09-13, 12:37–13:0x UTC.** Read at HEAD `e154dde`, clean
> tree. The builder is awake and between slots (last iteration ended 12:25,
> `rc=0`; the 13:07 slot has not fired). 22 iterations in 24 h, 21 `rc=0`, one
> `rc=124` at 09:57. Today is **Sunday**: the Review's FULL sitting ran at
> 06:37–07:0x and `docs/PROGRESS.md` carries it.

## VERDICT: DRIFTING

**Say the clean part first, because it is most of the audit and it is true.**
Sections 1, 2, 5 (accounting), 6 and 7 are clean and I checked them
mechanically. Zero thresholds moved in the loosening direction in 7 days —
every constant that moved moved *harder* (`MIN_DISTRACTOR_EVAL` 9→12,
`N_PROPERTIES` 15→16→17→18, `MAX_HELDOUT_CV_PCT` and `MAX_SPREAD_RATIO` armed
as new conjuncts). 372 distinct commits referenced by the ledger, **zero
dangling**. 106 PASS rows, **zero** without an implementation file, **zero**
without a declared control. Every open decision is armed; `UNDECLARED`,
`MEANS-ESCALATED` and `OVERDUE` are all empty. The builder refused three
dispatches today it was formally entitled to make and was right each time.
**The ledger is trustworthy. Nothing below is misconduct.**

The finding is about the **schedule** and the **frontier**, and it is one
finding wearing two faces:

**Fourteen queue rows were dated onto today. Today was their sitting. The
sitting happened at 06:37 and did not contain them** — and six of the fourteen
are the repair path for the specs that block this ladder, for three of the four
claim-dead GOAL.md commitments, and for both of the red terms that make
`coverage` exit 2.

Meanwhile: **in seven days not one spec on this ladder achieved a first-ever
PASS about Jack**, and `demonstrated` went 108 → 106 in the last 24 h. Both
demotions were honest. That is the problem — the honest work has nowhere left
to go, and the place it would come from is the desk that missed its sitting.

---

## RANK 1 — the fourteen rows due today are one cluster with one root, the root's sitting was this morning, and the FULL page does not mention it

**The numbers, from the tools.**

`run review-queue` prints, today:

```
2026-09-13  14  ##############  !! AMBER: pile
14 rows share 2026-09-13 against a measured capacity of 6/cycle
0 violations.
drain  UNBOUNDED — the desk is not keeping up. 45 live rows,
       arrivals exceed disposals by 7 over the window.
```

I re-derived the 14 by hand against `docs/REVIEW_QUEUE.md` (last-`DUE:`
semantics, live rows only) and get exactly the tool's list:

| status | row |
|---|---|
| OPEN | `ba03-null-saturates-the-horizon` |
| DISPOSITIONED | `cross-organ-doc-race-voids-certificates` |
| DISPOSITIONED | `lc07-checkpoint-branch` |
| DISPOSITIONED | `lt01-c2-body-cannot-rise` |
| OPEN | `sh02-null-saturation` |
| OPEN | `t205-world-model-loses-to-the-ridge-reference` |
| OPEN | `t215-heldout-language-routing-diagnosis-is-filed-behind-a-pilot-blocked-wall` |
| DISPOSITIONED | `t215-router-under-lexical-null` |
| OPEN | `t306-matched-magnitude-noise-buys-coverage` |
| OPEN | `t402-touch-drowns-audio-at-the-fusion-boundary` |
| OPEN | `two-eyes-one-certified` |
| DISPOSITIONED | `w0-too-shallow` |
| OPEN | `w1-world-edit-window` |
| OPEN | `xl01-death-and-retry-has-no-reachable-repair-path` |

**The consumer's turn has been taken.** `scripts/review.sh` is cron `37 6 * * *`;
it ran today 06:37–07:0x as the sixth Sunday FULL. I diffed
`docs/REVIEW_QUEUE.md` across every commit from `03e68bc` (the 92nd audit's read
at 06:25) to HEAD:

```
+ROUTED: pass-certificates-are-not-re-evaluated-when-a-dependency-falls | ... | OPEN
+ROUTED: completeness-audit-2026-09-13-the-cognitive-half-is-the-hole   | ... | OPEN
+ROUTED: t108-noise-floor-is-quoted-by-nobody                           | ... | OPEN
+ROUTED: t108-bar-set-from-n1-is-now-the-projects-largest-blocker       | ... | OPEN
```

**Four rows arrived. Zero rows were dispositioned, ACTED or DECLINED.** The
`review_queue_net_arrivals` ratchet moved `+4` on the same fact and is flagged
`!! MOVED` in `run status` right now.

**This is not fourteen independent promises — it is one cluster with one root,
and the root is `W1`.** From the row bodies themselves:

- `w1-world-edit-window` (OPEN, DUE 09-13) *is* the root: *"the single
  world-edit sitting that `W1.03` opens, which pays the 21-certificate
  `playground.py` mechanical bill ONCE."* It is itself `BLOCKED-BY:
  w0-too-shallow` (DISPOSITIONED, also DUE 09-13).
- `ne01-occlusion-knife-edge` (HELD, 20 d) and `water-apply-phantom-force`
  (HELD, 20 d) both declare `BLOCKED-BY: w1-world-edit-window`.
- `sh02-null-saturation` and `ba03-null-saturates-the-horizon` were each
  **re-armed twice** with the reason stated in the row: *"it should be picked
  IN LIGHT OF the w0-too-shallow design."*
- `lt01-c2-body-cannot-rise` was opened as its own row *"because both turn on
  the identical fork (is the repair the world, or the body?)."*

Six of the fourteen were deliberately walked onto this Sunday to be decided
together, by authors who said so in writing.

**And `W1` is absent from today's page.** `grep -nE "W1|w0-too-shallow|world-edit"
docs/PROGRESS.md` returns **nothing**. The FULL spent its sitting on Part 2 (the
T6.03 finding — real, and a good catch) and on routing four new rows. The page
does not record that the W1 sitting was owed today and did not happen.

**What it costs, mechanically.** These are not administrative rows:

| row | what it gates |
|---|---|
| `w1-world-edit-window` → NE.01 | `run blocked` #3 terminal blocker, **frees 7**, impl unchanged **19 d** |
| `lt01-c2-body-cannot-rise` → LT.01 | `run blocked` #2 terminal blocker, **frees 7**, impl unchanged **12 d** |
| `sh02-null-saturation` → SH.02 | claim-dead **thermal (kills)** *and* **shelter/building** |
| `ba03-null-saturates-the-horizon` → BA.03 | claim-dead **balance** |
| `lc07-checkpoint-branch` → LC.07 | welds GEN.02/03/06/09 = `coverage`'s 4 NEW-unrunnable citations; and the Learning-core seat's whole TRIGGER DEBT |
| `xl01-…-no-reachable-repair-path` → XL.01 | *"He lives, he dies, he remembers"* — FAIL since 2026-08-19 |

**Three of `coverage`'s four claim-dead commitments, and both of the red terms
that make `coverage` exit 2, have their repair in this pile.**

**The timing.** Rows go OVERDUE at 00:00 — about **11 hours** from now. The next
consumer cycle is 06:37 tomorrow, **a DAILY, not a FULL**, and it is the same
morning `D19` and `D25`'s defaults fire. Best measured capacity is 6 dated rows
per cycle. So the arithmetic is: **14 OVERDUE tomorrow, ≥8 still broken after
the desk's best possible day.** `review_queue_violations` has read 0 since
2026-09-03 and would take the largest single jump any ratchet in this repo has
ever taken.

**Why no instrument caught it, and this is the durable part.** `review_queue.py`
prints the pile histogram as AMBER and prints `0 violations` — both true. It has
no *forward* reading. It cannot say *"14 dated rows fall due before the next
consumer cycle against a measured capacity of 6."* It knows every number in that
sentence. Two consecutive overseers found the pile by reading a histogram by
eye; the 92nd audit named it at 06:25 as a forecast, the sitting passed twelve
minutes later, and nothing printed the transition from forecast to fact. It also
cannot see the *cluster* — the six rows' shared root lives in prose (`"in light
of the w0-too-shallow design"`), and `review_queue.py` reads declared fields
only, on purpose, because `champions.py` learned on `901f7fc` what a regex over
prose costs.

**Fairness.** The desk is not idle and is not lying. It disposed of 6 and
designed 14 this week while the builder was switched off for 4.3 of 7 days by
the shared usage meter; its FULL run has died at the wall on four of four
Sundays; and `D25`, open on the owner's desk with `decide_by` today, is *about*
that wall. The desk is over-subscribed, which is a capacity fact, not a conduct
fact. What is a finding is that nothing printed the collision, so nobody could
have chosen differently this morning.

---

## RANK 2 — the frontier: seven days, zero new capabilities, and there is nothing legal left to run

This is the honest answer to section 8, and it is a fact about the *ladder*, not
about the builder's effort.

Computed from `experiments/ledger.json` history, first-ever-PASS per spec:

- **Last 24 h:** 25 ledger events, 18 of them PASS. Specs achieving a
  **first-ever PASS: 0**. The only two first-ever *events* were `PL.02` VOID
  (00:17) and `LG.12` FAIL (05:15).
- **Last 7 days:** specs achieving a first-ever PASS: **2** — `T0.35`
  (2026-09-06) and `PL.00` (2026-09-07). Both are harness/rule specs;
  `coverage` lists `PL.00` under *"support passing, not credited"*.
- **`demonstrated` 108 → 106** in 24 h (`T6.03` BLOCKED at 06:44, `T1.08` FAIL
  at 10:05), against a registry that grew 245 → 246.

`run status` says it plainly: **96 re-buys across 45 specs**, *"instrument-coupled
53 of 107 (50%) — this project's own tool edits are what staled its
certificate."*

**The builder is not avoiding science; there is none available to it.**
`coverage`'s queue depth reads **4 dispatchable today, all 4 VOID → 0 FRESH
dispatch**, with four cost classes carrying `NOT FILLABLE: pilot BLOCKED on
evidence … the repair is a REDESIGN` and `cpu<10min` FILL-HELD behind `D19`. The
12:07 iteration enumerated all 43 runnable non-PASS specs by hand and reported
*"not one is fresh science."* I spot-checked that against `run blocked` and it
holds.

So the loop spends its hours on the instrument layer — the firing-diff check,
`run blast-radius`, `T0.28` strengthened 15→18 properties in a day. All of it
traces to GOAL.md's first principle (*"protects the honesty of watching what
happens when the three meet"*), all of it is well made, and **none of it is
drift**. But it is instruments watching instruments, and the exit from that loop
is not another instrument: it is the six rows in RANK 1.

The two ranks are the same finding. **The ladder's frontier now runs through the
review desk**, and the review desk's drain is `UNBOUNDED`.

---

## RANK 3 — `D1.0`'s 33.78 GPU-hours still have no verdict, and this morning its path to one got longer

`gpu_hours_no_verdict` from `run status`:

```
TOTAL 48.42 h
  D1.0          33.78 h / 2 attempt(s) / 0 verdict(s)   ← 70% of all GPU spend
  UNATTRIBUTED   6.32 h / 21 job(s)                     (at floor 21)
```

**`D1.0` is 70% of every GPU hour this project has ever spent and has produced
no ledger verdict.** It was ordered for today by `PROGRESS.md`'s item 3, and at
10:05 `T1.08` FAILed, putting `T1.08` between `D1.0` and any dispatch —
`run blocked` now ranks `T1.08` first at frees 41 / blocks 45, with `D1.0`
inside the blocked set. `run_spec` refused the dispatch. **That refusal was
correct** — the 92nd audit's B1 guard met its first real dispatch and held — and
I want it on the record as a success, not a complaint. But the consequence is
that the largest single compute expenditure in the project is now verdictless
behind a FAIL whose repair is a design question routed **DUE 09-16**.

GPU weeks are clean otherwise: W37 opened today, **0.82 h of 30 used** (2 jobs),
and `T1.08`'s 0.36 h bought a real FAIL — a verdict, honestly harvested.
`gpu_unattributed_jobs` = 21, AT floor.

---

## The audit, section by section

### 1. Integrity of the ledger — CLEAN

Checked mechanically over all 145 rows / 106 PASS:

| check | result |
|---|---|
| commits still resolve (372 distinct, incl. history) | **0 dangling** |
| PASS with no implementation in `experiments/tests/` | **0** |
| PASS with no `control` declared | **0** |
| PASS stamped `+dirty` | **1 — `T0.23`, and it is benign (below)** |

`T0.23` PASS records `b4f123d+dirty` (attempt 11, 10:18:13, 37.11 s), acquired
*after* the 92nd audit correctly reported no dirty stamps. I chased it and the
repo's own instrument had already answered it. `run stale`:

> `T0.23 PASS ran from a modified tree at b4f123d, but the implementation it
> names IS committed: impl_sha bf80bba7f5e8 reconstructs byte-identically at
> 8821ff14 … the uncommitted edits were outside what this spec declares (1
> uncommitted code file(s) at run time, none of them this spec's own).`

The uncommitted file was the `run blast-radius` CLI edit landed minutes later as
`8f3b52a`. Per `protocol.py:178`, a single-spec run from a dirty tree *"only
fails to certify, which is honest and normal"*; the gate-run hazard that cost
`T0.09` three hours does not apply. **Cosmetic, correctly characterised by the
tool, no action needed.** `run stale` also flags `T6.03` (BLOCKED, pre-`impl_sha`,
own test file uncommitted) — not a live claim, and already owned by
`pass-certificates-are-not-re-evaluated-when-a-dependency-falls` (DUE 09-16).

`T0.01` and `T0.10` implementations contain no literal `control` token; both are
Tier-0 fixtures (`Repo imports clean`, `Kaggle job round-trip`) whose specs
declare controls exercised through the harness. Not a finding.

### 2. Thresholds and controls over 7 days — CLEAN, and I looked hard

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`, filtered for constant
moves, deleted constants, `seeds=` reductions, removed `control=`, and `or`
added to `_check`.

- **Constants deleted: one.** `MIN_DISTRACTOR_EVAL = 9` in
  `me_9_attributed_recall.py`, commit `4f62755`, replaced by **12** — a
  strengthening, with the arithmetic at the constant (*"9 tolerated the control
  going two-thirds quiet for no stated reason; 12 is 80% of the pool"*) and the
  recorded measurement (15/15) clearing it with margin.
- **`MIN_PARA_MARGIN 0.10 → 0.05`** appears in the diff and is **not an edit** —
  it is a planted-diff string inside the firing-diff check's own red-first
  fixture. I verified it is inside a quoted literal.
- `seeds=` changes: two, both on newly-registered specs. `control=` removals:
  **none**.
- Ratchet floors: `UNREACHABLE_BASELINE` grew **94 → 97** today (`deda088`).
  Growth of a shrink-only floor is the exact shape I am here to catch, so I read
  it in full: `T1.08` FAILed, and `D1.0`/`T2.01`/`T2.02` all declare
  `depends_on: T1.08` and left the reachable set mechanically. The commit
  message carries the growth-log entry and the sentence *"the growth is MINE."*
  **Justified by a measurement, correctly attributed, not a loosening.**

**No silent loosening. Zero findings.**

### 3. Drift from the goal — no drift; the converse is RANK 2

Last 24 h of builder work, each against a GOAL.md sentence:

| work | sentence served |
|---|---|
| firing-diff check (`a662fab`, `7fbe18c`) | first principle: *"protects the honesty of watching…"* |
| `run blast-radius` (`8f3b52a`) | same — pricing a gate edit before making it |
| `T1.08` harvest (`deda088`) | *"every capability claimed only by an experiment that could have failed"* |
| `T0.28` 15→18 properties | same |
| `LG.12` registered (`939e3c4`) | *"The LLM is his mouth, never his mind"* |

**Nothing serves no sentence.** The converse question — which parts of GOAL.md
have no passing spec — is RANK 2 and `coverage`'s own output: 4 CLAIM-DEAD
commitments (smell, balance, thermal, shelter/building), 9 more with live claim
specs and nothing passing, including **curiosity (2 pass of 12)**,
**one brain / unison (1 of 27)**, **fast/slow (0 of 8)**, **sleep (0 of 5)**.

### 4. Builder liveness — HEALTHY

22 iterations in 24 h; **21 `rc=0`**, one `rc=124` (09:07 slot, wall clock at
09:57, work picked up cleanly by the 10:07 slot). ~40 commits. No pause, no
credit exhaustion, no repeated identical failure. `week:all models` **82%**
against a pace line of 83.84 — under the 90% stop, and the builder is correctly
reading the all-models line rather than the pinned `week:Fable 100%`. PASS delta
is **−2**, both demotions honest and both louder than a PASS would have been.

### 5. Compute honesty — accounting CLEAN, one standing waste (RANK 3)

GPU by week: W32 16.61 h, W33 7.89, W34 1.62, W35 19.20, W36 17.73, **W37 0.82
of 30** (opened today). Kaggle resets Sunday; **29.18 h remain this week** and
there is currently nothing legal to spend them on (`D1.0` foreclosed behind
`T1.08`). CPU day meter is itemised per spec and billing re-runs correctly
(`9ceefb8` bills `T0.23`'s 37.11 s as the receipt for `8f3b52a`).

### 6. Stuck decisions — CLEAN

`decisions --check` → `EXIT 0`, `ratchet ok (0/10 undeclared, 0/3
unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0 default-action-expired, 0/0
firing-diff)`. **No `MEANS-ESCALATED`. No `UNDECLARED` to arm** — the ratchet is
at floor, so I have none to arm this audit. **No `OVERDUE`.** Four open, all
armed: `D19` (09-14), `D20` (09-18), `D25` (09-13), `D27` (09-20).

I checked the converse — decisions acted on without being recorded — across
`D18`, `D22`, `D23`, `D24`, `D26`: every one carries a `RESOLVED BY ARMED
DEFAULT` entry with the required sentence *"the owner did not rule by <date>, so
the pre-registered default fired"* and a reversal path. `D25` was **correctly
not fired** today: `decisions.py:222` marks overdue at `(today - decide_by).days
> 0`, so the earliest legal firing is 09-14. The builder caught and documented
this against two pages that told it to fire today. That is the right call.

### 7. Bakeoff hygiene — CLEAN

`DECISIONS_RESOLVED.md` read for verdicts inside the noise margin, VOIDs treated
as verdicts, and gate-free decisions. The one decision made without the 3σ
learning gate — **PL.00/RENDER → coarse-shadow512** — declares exactly why on its
face: *"probe, not run_bakeoff — the arms are loop configurations, not learners,
so the 3-sigma learning gate has no referent; the probe carries PL.00's own VOID
gates instead."* Declared and reasoned, not smuggled. `D10`'s wm-latent seat is
held BY VERDICT off a VOID and says so on its face, with the single-arm caveat
attached; `champions --check` counts it in `UNVERIFIED VERDICTS` (2/2, at floor).

### 8. Are we closer? — No, and the reason is legible

Not closer this week. Two specs achieved a first-ever PASS in seven days, both
harness; **zero** in twenty-four hours; `demonstrated` fell 108 → 106. And this
is *not* the "longer list of green ticks" failure — the list got **shorter**,
honestly, twice, because instruments the builder sharpened found two
certificates that were not owed. That is the ladder working.

What is not working is that the ladder has run out of legal moves. Four cost
classes read `NOT FILLABLE`, the fresh-dispatch queue reads **0**, and every
exit from that state — `W1`, `LT.01`'s body-vs-world fork, `SH.02`, `BA.03`,
`LC.07` — is a design decision sitting on one desk whose drain is `UNBOUNDED`,
dated onto a Sunday that has now passed.

`coverage` exits **2**, as it has for days, on `claim_dead=4` and
`new_unrunnable_citation=4`. I verified both are honestly owned
(`five-commitments-are-claim-dead-behind-foreclosures` DUE 09-16;
`goal-cites-four-specs-that-resolve-to-corpses` DUE 09-15) and that the standing
red cannot silence anything — the 64th audit's ratchet counters print every
number in `run status` regardless. **That mitigation works; I checked it rather
than assuming it.** But it is worth saying out loud that a tool which has exited
2 for a fortnight is a tool whose exit code has stopped carrying information,
and `T0.23`'s own certificate now records `experiments.coverage::bare → EXIT 2,
ok: True` as an *expected* value. That is the `lib_seal.sh` disease `D25` was
escalated about, growing in a second organ.

---

## FOR THE BUILDER

**B1 (today, before 00:00 UTC — the only time-critical item). Act on the one row
in the fourteen whose remaining debt is yours, and make the collision visible on
the rest.** `lc07-checkpoint-branch`'s body says the residue explicitly: *"What
remains owed on this row is the ONE cheap thing: the BUILDER prices the CPU
venue — 526 GPU-wall-hours through the pilot's own measured dec/s."* That is a
CPU-cheap arithmetic job, it is yours, and it is due today. Do it and stamp
`ACTED` with the commit.

For the other thirteen: **annotate, do not re-date.** Follow your own precedent
in `70e2686` (91st audit B4), where you flagged an expired premise and
deliberately left the date alone. Add one line to each of the six cluster rows
(`w1-world-edit-window`, `w0-too-shallow`, `sh02-null-saturation`,
`ba03-null-saturates-the-horizon`, `lt01-c2-body-cannot-rise`, plus the two HELD
rows behind them) recording the fact, not a new promise: *the 09-13 FULL sitting
these rows were dated to did not take up W1.* Re-dating the Review's design debt
is the Review's call, not yours; recording that a sitting passed is bookkeeping,
and right now nothing in the repo records it.

**B2 (the durable repair — build this). `review_queue.py` gains a forward
reading.** It already computes every term and can only say `0 violations` until
after the promises break. Add an `IMMINENT` reading in the idiom of the existing
counters:

> `IMMINENT — N live dated row(s) fall due before the next consumer cycle
> (<date>), against a measured capacity of M/cycle. K of them cannot be
> discharged by that cycle.`

Derive the next consumer cycle from the same git history the throughput block
already reads. **Reporting-only and unfloored** — it must not gate, because a
pile is a legal state and a gate here would forbid a legal move, exactly as the
`piled_on` metric already reasons. Red-first it against a fixture where the pile
exceeds capacity and one where it does not. Today's reading would have been
`IMMINENT 14 against 6, 8 undischargeable`, printed at 06:25 instead of found by
eye at 12:40.

**B3 (smaller, same family). The cluster is invisible because it lives in
prose.** Six of the fourteen share one root and say so only in body text
(`"in light of the w0-too-shallow design"`), which `review_queue.py` will never
read and should never read. The declared-field repair, in the `DUE:`/`COVERS:`/
`BLOCKED-BY:` idiom: allow **`WAITS-ON: <row id>`** on live non-`HELD` rows —
`BLOCKED-BY:` already exists but buys ageing-exemption, which is wrong here
because these rows *should* age. Then the pile histogram can print *"14 rows, 6
of them behind one root"*, which is a different and much more actionable
sentence than *"14 rows."* Propose it; do not implement it ahead of B2.

**B4. Record the moved ratchets.** `run status` currently flags three `!! MOVED`
readings — `review_queue_net_arrivals` 3→7, `fail_unowned_owned_forms`
queue-row 21→22, `gpu_hours_no_verdict` TOTAL 48.07→48.42 h (the new `T1.08`
line). All three moved for committed acts that are justified (today's four
routings, `T1.08`'s dispatch). `run ratchets record` them so the next reader
does not have to re-derive which acts caused them, as I just did.

---

## FOR THE OWNER

**1. Tomorrow morning is crowded, and you should know before it happens.** At
00:00 on 2026-09-14, fourteen queue rows go OVERDUE. At 06:37 the Review runs —
a **DAILY**, not a FULL — and on the same morning `D19` and `D25`'s armed
defaults become fireable. The desk's best measured cycle discharges 6 dated
rows. Nothing here needs your permission; I am telling you because
`review_queue_violations` has read 0 since 2026-09-03 and will not read 0 again,
and you should see the cause rather than the jump.

**2. `D25` decides today and its answer is the first lever on all of this.** Its
default (iii) — FIX THE SEAL, BUY NOTHING — is the only legal one of the three
and I endorse it. But note what the decision is *about*: the Review's FULL run
has died at the wall on **four of four** Sundays, and option (i) RAISE THE WALL
CLOCK is excluded from the default only because a default may not spend your
credits by silence. **That exclusion is correct and it is also the binding
constraint on RANK 1.** The desk missing the W1 sitting is downstream of a
sitting that keeps running out of time. If you want one thing to rule on today,
rule on `D25` — and if you want to rule wider than the seal, (i) is yours to
take and nobody else's.

**3. The structural question, which I am deliberately NOT opening as a new
decision.** The ladder's frontier now runs through a single desk with 45 live
rows, a measured capacity of 6 dated rows per cycle, and a drain the tool calls
`UNBOUNDED`. Four cost classes are `NOT FILLABLE`, the fresh-dispatch queue is
**0**, and no spec has achieved a first-ever PASS in seven days. The levers are
yours and they are not equivalent: more Review wall-clock (`D25` option (i)),
fewer rows routed, or a different split of design authority (`D22` fired on
2026-09-12 keeping design authority with the Review, unchanged and unnarrowed).

I am not filing this as `D28`. You have four open decisions and `D25` is
literally about this desk's clock; a fifth entry competing with it would be
noise, and the honest sequencing is to see what `D25` buys before asking for
more. If tomorrow's DAILY discharges fewer than 6 of the fourteen, the next
audit should escalate it formally and I have said so here so that the decision
not to escalate today is on the record rather than implied.

**4. `D27` already carries my predecessor's recommendation** — (i) BUILD THE
SCREEN, reporting-only until its false-positive rate is measured — with the
prototype's 104-of-107 false-positive rate published on the entry rather than
buried. `decide_by` 2026-09-20. Nothing from me to add; the entry is honest.

---

*93rd audit. Instruments: `coverage` EXIT 2 (claim_dead 4, new_unrunnable_citation 4 — both owned, both routed), `decisions --check` EXIT 0 at floor, `champions --check` EXIT 0 at floor, `review-queue` EXIT 0 / 0 violations / AMBER pile 14-on-6, `run stale` 3 modified-tree claims (1 live PASS, benign), `run status` 106/246.*
