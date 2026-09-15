# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **97th audit — 2026-09-15, 06:37–07:0x UTC.** Read at HEAD `008f2eb`, clean
> tree. **The builder is NOT between slots. It has been dark for 19 consecutive
> hours.** Last productive iteration ended `rc=0` at **2026-09-14T11:14:44**;
> every slot since — **19 of them**, 12:07 on 09-14 through 06:07 today — was a
> `PACING` skip. `demonstrated` **108 → 108**, flat since 2026-09-14T04:18.
> **One commit** exists in those 19 hours and it is the loop's own mechanical
> bookkeeping commit.

## VERDICT: DRIFTING

Not `INTEGRITY RISK`: I went looking for a rotten claim and did not find one.
No threshold moved in the loosening direction in seven days, every PASS resolves
to a spec and a live commit, the one dirty certificate is classified by an
instrument that already exists, and `DECISIONS_RESOLVED.md` contains a bakeoff
(`SO.10`) that is the best-handled tie this project has recorded. The ledger is
honest.

It is `DRIFTING` because the project stopped. For nineteen hours the system has
had a paid-for GPU result lying unread on disk with its answer already in it, a
pre-registered deadline quietly slipping past its own firing date, and four
owner-named constitutional commitments claim-dead for twelve days. **Nothing is
false. Nothing is moving.**

---

## THE FOUR INSTRUMENTS

| instrument | exit | reading |
|---|---|---|
| `coverage` | 2 | **0 commitments with NO declared spec.** 4 CLAIM-DEAD, 9 live-but-nothing-passing. Ratchet classes at floor. |
| `decisions --check` | 0 | ratchet ok (0/10 undeclared, 0/3 unrouted-owner-ask, 0/0 vanished, 0/0 default-action-expired). **But `D19` prints `OVERDUE — DEFAULT IS DUE TO FIRE` — see RANK 2.** |
| `champions --check` | 0 | 0 phantom arenas, 2/3 unfalsifiable, 4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger debt — all AT their declared counts. |
| `run review-queue` | 2 | **6 OVERDUE**, ratchet `review_queue_violations` **0 → 6**. See RANK 4. |

No `MEANS-ESCALATED`. No `ARENA-MISSING`. No `UNDECLARED` decision. No
`UNROUTED-OWNER-ASK` against `PROGRESS.md`. The three instruments this organ
usually has to argue with are clean, and I am saying so rather than inventing a
reason they should not be.

---

## RANK 1 — A finished GPU result has sat unread for 19 hours, and its answer is already on disk

This is the finding. Everything else is smaller.

**The facts, from disk and not from anyone's report:**

- `/data/t108_backend_probe.json`, mtime **2026-09-14 11:22**. Nineteen hours
  and twenty minutes before I read it.
- The 10:07 slot relaunched the colab arm detached (pid 405151) and wrote into
  its journal: *"The next iteration harvests... if colab failed again, the new
  `stdout_head` in `failures[]` decides cause-1 (multi-dir fetch) vs cause-2
  (structurally dead colab lane)."*
- pid 405151 is **DEAD** — it exited at 11:22, 45 minutes before the first pace
  skip. It is still listed in `/data/jack-logs/declared_pids` as a live
  declaration, twenty hours later.
- **The instrumentation worked and the question is answered.** The failure
  record's second entry reads
  `stdout_head='JACK_OUT /content\nREPO 521d33e\n...'`. The job wrote its
  artifact to `/content`; the fetch looked in `/content`; the fetch returned
  *File or directory not found*. That is **cause-2**, not cause-1 — and the
  builder pre-registered that read before the number existed.
- The 1.08 GPU-h that bought this diagnostic was spent **specifically** so this
  sentence could be written from disk instead of argued. It has not been read.

**And there is more in the record than anyone has looked at.** The
`stdout_tail` of both failed colab attempts contains the arm's actual per-seed
numbers — `seed 2 heldout 0.04714763`, `seed 3 heldout 0.09833387`,
`seed 4 heldout 0.03536733` — byte-identical across the two attempts, which is
what a deterministic kernel on fixed seeds should look like. I am not doing the
science and I am not proposing the branch read; that is the builder's unit and
the ruling pre-registered how it is taken. I am reporting that **the arm the
project has twice called "lost" is partially present in the failure record the
builder built to catch exactly this.**

**Why nobody saw it — and this is the structural half.** `scripts/ladder_loop.sh`
line 112 states the governing principle in its own words:

> `# A PACE SKIP DEFERS CLAUDE SPEND — IT MUST NOT DEFER WORK THAT COSTS NONE.`

It then implements that principle for exactly **one** class of zero-cost work:
committing a detached run's ledger row and receipts. `HARVEST_PATHS` is
`ledger.json gpu_budget.json gpu_submissions.jsonl cpu_budget.json` — four files
*inside the repo*. The T1.08 probe, by design and correctly, writes its result
to `/data/t108_backend_probe.json`, **outside the repo**, and buys no ledger row.

So commit `008f2eb` — the only commit in nineteen hours — **committed the
probe's bill and left its answer on the floor.** Its own message says
`detached-run ledger row(s) [unknown]`, and `[unknown]` is not a bug: `ROWS`
computed to "unknown" because no ledger row changed. Only the receipts did. The
mechanical commit correctly reported that it did not know what it was
committing, and the thing it was not committing was the science.

---

## RANK 2 — `D19`'s armed default was due to fire at 00:00 today and did not. A deadline conditional on a spend meter is not a deadline.

`decisions --check` prints `D19 costs 3 specs OVERDUE — DEFAULT IS DUE TO FIRE`
(blocks HR.1–HR.4, class `goal`, default NO FETCH, `decide_by` 2026-09-14). The
Review ordered it in `PROGRESS.md` FOR THE BUILDER item 3 with required wording,
firable from **2026-09-15 00:00**. Seven pace-skipped slots have passed since
midnight. It has not fired.

**The control case is in this repo and it is one day old.** `D25` fired by armed
default at **2026-09-14 ~00:2x**, on a night the builder was awake. `D19` was due
at 00:00 on a night it was pace-gated. Same mechanism, same hour of the day,
opposite outcome, and the only variable is a usage meter.

This matters more than one late decision. The armed-default machinery exists
*because* `D1` sat OPEN for twenty days blocking 38 specs. Deadlines replaced
deadlock. **A deadline that only fires when the builder happens to be unpaced
has an unadvertised dependency on a budget meter — and 65% of that meter is
spent by work that is not this project** (`usage_attribution`: of this week's 26
shared points, builder 7, desks 2, NOT THIS PROJECT 17). The deadlock came back
through a side door, and no instrument reports the coupling, because the organ
that would report it is the organ being skipped.

Firing a default costs one commit and zero meter. It belongs on the skip path,
next to the harvest, under the comment already written there.

**Release estimate, so the next reader is not guessing:** `allow` is a pure
function of the clock at 0.3869 pts/h. At 06:07 today it read 35 against usage
37. The builder releases at roughly **13:00–14:00 UTC today** — call it a
**26-hour blackout** — and later still if the other 65% of the meter keeps
drawing. `D19` fires ~14 hours late; the probe result goes unread for ~27 hours.

---

## RANK 3 — 2.67 GPU-h spent against a 1.20 h authorisation, and no instrument in this repo can print that sentence

**Section 5, and it is a structural blind spot, not a one-off.**

The Review's 09-14 ruling AUTHORISED the T1.08 backend-confound probe at
**n=5 per backend, 1.20 GPU-h**. Actual spend, from `gpu_budget.json` and
`gpu_submissions.jsonl`:

| job | backend | hours | outcome |
|---|---|---|---|
| `jack-ladder-1789370135` | kaggle | 0.5607 | landed, usable (cv 36.577) |
| `ladder-1789373334` | colab | 1.0277 | computed; artifact not retrieved |
| `ladder-1789381054` | colab | 1.0832 | computed; artifact not retrieved |
| **total** | | **2.6716** | **223% of the authorisation; the colab arm still has no committed reading** |

Both colab attempts also declared `est_hours: 0.7` and billed 1.03 / 1.08 — 47%
and 55% over their own declared estimates, each.

`"overruns": []`. Nothing is red. Three reasons, all in the code:

1. **The overrun mark is gated on one backend.** `gpu.py:521` —
   `if backend == "kaggle" and used > KAGGLE_WEEKLY_HOURS`. Colab never marks.
2. **Colab has no ceiling at all.** `gpu.py:424` — `remaining()` returns
   `float("inf")` for any backend that is not kaggle, so `afford("colab", h)` is
   **always True**. Colab hours are billed to the week counter and constrain
   nothing. W37 colab stands at 2.1109 h, 100% of it on these two unretrieved
   attempts. The builder's retry path prefers colab. **Nothing in this repository
   would have stopped a tenth attempt.**
3. **A PROBE is structurally invisible to the counter built to catch this.**
   `gpu_hours_no_verdict` joins charged jobs against `ledger.results`
   (`run.py:1456`). Its own docstring says it exists because *"D1.0 bought 33.78
   GPU-hours across two attempts for zero verdicts and no instrument could print
   that sentence."* But the ruling defines a probe as buying **no ledger row** —
   correctly, that is what keeps a probe from purchasing a verdict. The
   consequence nobody costed: probe hours can never join, so the one counter
   that measures GPU-with-nothing-to-show-for-it cannot see the one class of
   spend that is *guaranteed* to have no ledger row. `run status` still reads
   `T1.08: 0.36 h / 1 attempt / 1 verdict`.

`afford()` gates a single job against a weekly ceiling. **A spec that overruns by
retrying passes `afford()` every time**, because each retry is small on its own.
That is precisely what happened.

W37 standing: kaggle **1.379 / 30**, 28.6 h free, expires Sat 09-19. Colab
2.1109, unmetered.

---

## RANK 4 — 6 OVERDUE queue rows; the ratchet moved 0 → 6

`review_queue_violations = 6  !! MOVED +6 since 2026-09-03`. All six promised
**2026-09-14** and broke at midnight: `me1-similarity-floor-never-abstains`,
`d10-successor-rerun-under-adopted-gate`,
`lg03-blind-twin-cannot-prove-itself-alive`,
`so07-recording-worlds-fail-the-reference-bar`,
`pl02-eye-gate-reads-the-encoder-not-the-eye`,
`lg12-abstention-knob-has-no-resolution`.

I am ranking this fourth rather than first, and the reason is that the Review
**forecast this in writing before it happened, in the open, against itself.**
`PROGRESS.md` FOR THE OWNER item 1 re-armed thirteen rows on 09-14, recorded
`EXIT 0, 0 violations` at that moment, bound itself to a stop-rule — *"if the
fourth breaks, they are DECLINED as a class"* — and published the honest evidence
against its own `D28` default. Six of thirteen broke within a day. That is a desk
failing to keep up, declared in advance, which is a different and far less
serious thing than a desk hiding a backlog.

Two readings the tool gives that the desk should carry into its next sitting:
`DUE-DATE PILE` shows 09-15, 09-16, 09-17 and 09-20 each carrying 6 against a
measured capacity of 6 — **four consecutive days at capacity with zero slack** —
and `next_free_due` is 2026-09-18. And the drain reads `UNBOUNDED`: 23 settled
FAILs whose only repair owner is a row on this desk.

The related structural point belongs in RANK 2's family: `D28`'s default (a)
OVERDUE FIRST is due 09-21, and the Review has itself recommended amending it
before it fires. Noted, not re-asked.

---

## RANK 5 — `declared_pids` still declares a process that exited twenty hours ago

```
405151:1230905420  2026-09-14T10:17:53+00:00  dispatch T1.08 probe (colab arm retry, head 521d33e)
```

`kill -0 405151` → no such process. It exited at 11:22 on 09-14. The entry has
outlived it by twenty hours and will be read by the next waking slot as evidence
that the probe is still in flight. Small, and worth one line because it is
exactly the trap the loop has already been burned by twice: *"waiting on
background work" is a claim, not evidence.* A stale declaration is a claim that
survives its own evidence.

---

## THE SECTIONS WHERE I FOUND NOTHING, AND THAT IS THE RESULT

**1. Integrity of the ledger — CLEAN.** 108 standing PASS rows, 148 ledger
entries. All 108 resolve to a spec in `BY_ID`. Of 57 distinct commits, **56 exist
in git**; the 57th is `b4f123d+dirty` (T0.23, attempt 11, 2026-09-13) — and that
is not an unreported hole: `dirty_recoverability` classifies dirty rows into
COMMITTED / PRESERVED / LOST / UNSTAMPED, `run.py:1988` records *"2026-09-13: one
live dirty row, COMMITTED"*, and the class is asserted against constructed probes
so the reporting path cannot silently stop working. The record is honest about
its own soft spot. Two PASS rows carry no `control_metrics` (T0.01, T0.10) and
both are the declared by-decision exceptions `coverage` already names.

**2. Thresholds and controls over time — CLEAN, and it is the good kind of
clean.** 40 commits touched `registry.py` / `registry_expansion.py` /
`tests/` in seven days. Every numeric movement I can find is a **tightening**:
`MIN_DISTRACTOR_EVAL` 30 → **59** in three ME specs and `N_DISTRACTOR` 60 → 130,
each justified by arithmetic stated in the commit (`a_L = γ^(1/m)`, so m ≥ 59 is
the γ=0.05 minimum that can certify 0.95 at all), with **the 0.95 bar untouched
in both directions** and the specs re-bought PASS afterward rather than
grandfathered. `T1.08`'s `MAX_HELDOUT_CV_PCT` 7.0 is imported by the probe rather
than copied, specifically so it cannot drift from what it diagnoses. No `_check`
gained an `or`. No control was deleted or weakened. No seed count was reduced —
`PROBE_SEEDS` *extends* the registered `SEEDS = [0,1,2]` to five, and branch
(ii)'s count was fixed at 20 **before any number existed**. Nothing to report,
and the discipline on display is better than the average week.

**6. Stuck decisions — nothing hidden.** `D20`, `D27`, `D28`, `D29` are armed
with defaults and future dates. `D29` is correctly on the owner's desk rather
than the loop's: it asks whether an ARCHITECTURE seat may hold `BY VERDICT` while
a guard its own governing document calls mandatory was never armed — and its
default reasoning explicitly refuses the option that would shrink
`champions --check`'s own counter by re-labelling. That is the `T0.31` lesson
being applied by an organ to itself, unprompted. No decision was quietly acted on
without being recorded.

**7. Bakeoff hygiene — CLEAN, and `SO.10` is the exemplar.** The machine named
`laplace-full` the winner at 5.79σ. The spec's *second* pre-registered gate
disqualified it — it cannot migrate, divergence negative on all three seeds
against `MIN_MIGRATE` 0.40 — so `SO.10` recorded **FAIL** and the Person-model
seat stayed **VACANT** rather than being handed to the best eligible arm after
the fact. The control (`pooled-scalar`) failed as a control must. The residual
question was **routed**, not decided. A tie at 0.26σ was called a tie. This is
what the whole apparatus is for, and it worked without anyone watching.

---

## 3. Drift from the goal

**What the builder did in its last working day, and what each serves — no
drift.** The T1.08 probe and T1.07's `IMPL_DEPS` declaration serve *"Really
learning, not appearing to learn"* — they are the falsification ladder
measuring its own noise floor and its own staleness, which is the honesty clause
rather than a capability. ME.1/ME.3/ME.5's denominators serve *"Memory makes it
him"*. The `FIELD_WATCH.md` reader serves the same honesty clause. Four units,
four GOAL.md sentences, zero drift. The builder also refused three dead orders
off a steering page rather than executing them, which is the behaviour this organ
has asked for repeatedly.

**The converse question is where the damage is.** `claim_dead = 4, unchanged
since 2026-09-03` — **twelve days**. The four are **smell**, **balance**,
**thermal (kills)** and **shelter/building**. Read them against GOAL.md in the
owner's own words:

- *"too cold kills him, too hot kills him"* — thermal, claim-dead.
- *"Cold nights teach shelter-building the way no scripted lesson can"* —
  shelter, claim-dead. `coverage` annotates this one **"owner's own image of
  success"**.
- *"EVERY SENSE A HUMAN HAS... Smell and taste are not ornaments"* — smell,
  claim-dead; balance, claim-dead.

Every park was legal and evidence-backed, and I am not asking for any PARKED
marker to be touched — deleting one would be strictly worse. But `coverage` is
right that leaving the commitment claim-dead is the bug, and three of these four
successors (`SH.02`, `SM.03`, `BA.03`) are redesigns owed by the Review desk that
is currently six promises overdue. **The commitments the owner described most
vividly are the ones with nothing runnable behind them, and they have been that
way for twelve days.** Add `PARK-ON-AN-UNREACHABLE-RELEASE = 3`: each of those
parks names a revival path that cannot be walked today.

Curiosity: 12 specs, **2 passing**, 5 nominated. One-brain/unison: 27 specs,
**1 passing**. Those are the two claims GOAL.md leans on hardest and they remain
the thinnest on the board.

---

## 4. Is the builder alive and productive?

**Alive, correctly gated, and stopped.** 19 `PACING` skips, 12:07 on 09-14
through 06:07 today. In the preceding 24 h: 7 iterations, 6 `rc=0`, one `rc=1`
(a 529 Overloaded at 09:07 on 09-14 — transient, server-side, not a loop fault).
`demonstrated` 108 → 108 across every one of them.

**The gate is not broken and I want to be precise about that.** The 90% stop is
the owner's rule; the pacing line is a self-imposed convergence to it; `allow` is
a pure function of the clock with zero variance and provably converges on 90, so
"pace_gate never releases the builder" is unavailable as a sentence. The gate is
doing its job. **The finding is the coupling, not the gate** — zero-cost,
time-critical work (an armed default's firing date; a finished detached run's
out-of-repo artifact) is riding on a spend meter, 65% of which is drawn by work
that is not this project.

---

## 8. The honest summary

**No. We are not closer than we were yesterday, and this time it is not even a
question about the quality of the green ticks.**

Yesterday I could have written the usual version of this paragraph — busy,
rigorous, not obviously moving. Today the answer is simpler and worse: for
nineteen of the last twenty hours this project did not run. One commit, written
by a shell script. `demonstrated` has not moved since 04:18 on 09-14.

What makes it sting is *what* was sitting there during the silence. A GPU result
that cost real hours, whose diagnostic instrumentation was purpose-built the same
morning to answer one specific question, landed on disk at 11:22 with the answer
in it — `JACK_OUT /content` — and forty-five minutes later the loop went quiet
and has not looked. The builder did everything right: it pre-registered the read,
it refused to guess at the cause, it instrumented instead, it wrote the harvest
instructions into its journal for its successor. Its successor never woke up.

And the pieces that *are* working are working beautifully in a way that makes the
stall harder to excuse, not easier. `SO.10` refused its own winner. `D29`'s
default refuses to buy itself a greener counter. The builder refused three dead
orders in a row and said why. The Review published the evidence against its own
armed default because it is the organ that default constrains. Every organ in
this system has learned to tell the others they are wrong. **None of them has
learned to notice that nothing is happening** — because the loop's only heartbeat
is the loop, and a paced-out loop reports its own silence in a log nobody reads
until an overseer runs.

Meanwhile "too cold kills him" and "he builds a shelter" — the two sentences the
owner used to describe what success *looks like* — have had nothing falsifiable
behind them for twelve days.

The ladder is the right ladder. The scoreboard is honest. The creature is not
getting more alive, and today the reason was not a hard problem. It was that
nobody was home.

---

## FOR THE BUILDER

1. **Harvest `/data/t108_backend_probe.json` as the first act of your next
   unpaced slot, before anything else.** It has been on disk since
   2026-09-14 11:22. `failures[1].stdout_head` reads `JACK_OUT /content` — the
   job wrote where the fetch looked, which is **cause-2** (kept-session
   retrieval), not cause-1 (wrong directory), and that is the read your own
   10:07 journal pre-registered. Take the branch the ruling pre-registered; do
   not re-derive it, and do not let a third colab attempt be the first thing you
   reach for. **Also read `failures[*].stdout_tail`** — three of the colab arm's
   five per-seed `heldout` values are in the record, byte-identical across both
   attempts. Whether that is enough to satisfy the ruling's n=5 is your call and
   the ruling's, not mine; I am reporting only that the data is not as lost as
   the last three journal entries assume.

2. **The pace-skip path must fire an overdue armed default.** `ladder_loop.sh`
   line 112 already states the rule — *"A PACE SKIP DEFERS CLAUDE SPEND — IT MUST
   NOT DEFER WORK THAT COSTS NONE"* — and implements it for exactly one class of
   zero-cost work. Firing a default is one commit and zero meter. `D19` was due
   at 00:00 today and has been skipped seven times; `D25` fired at 00:2x on 09-14
   only because the builder happened to be awake. Add the `decisions --check`
   OVERDUE read to the skip branch beside `harvest_bookkeeping`, and fire it
   there with the required wording. **Fire `D19` now, on your next slot,
   regardless** — *"the owner did not rule by 2026-09-14, so the pre-registered
   default fired"* — and record that it fired late and why.

3. **`harvest_bookkeeping` cannot see an out-of-repo artifact, and that is how
   the probe's answer was left on the floor while its bill was committed.**
   `HARVEST_PATHS` is four in-repo files; the probe writes to `/data/` by design
   and buys no ledger row, so `ROWS` computed to `[unknown]` in `008f2eb` — the
   commit correctly reported it did not know what it had committed. The repair is
   not to widen `git add` (the add -A ban stands). It is for the skip path to
   **detect and announce** a finished detached artifact — check `declared_pids`
   for an exited pid whose declaration names a dispatch, and `say` it into
   `ladder.log` loudly enough that the next waking slot cannot miss it. Committing
   the interpretation still belongs to an unpaced iteration; *noticing* costs
   nothing.

4. **`declared_pids` is not pruned on exit.** pid 405151 has been declared live
   for twenty hours and has been dead for twenty hours. Reap entries whose pid is
   gone, or stamp them `EXITED <ts>` — a declaration that outlives its process is
   a claim that outlived its evidence, and the next slot reads it as "work in
   flight".

5. **Colab is an unmetered lane.** `gpu.py:424` returns `float("inf")` from
   `remaining()` for every non-kaggle backend, so `afford("colab", h)` is always
   True; `gpu.py:521` gates the overrun mark on `backend == "kaggle"`. W37 colab
   is at 2.1109 h, 100% of it on two unretrieved attempts, and nothing would have
   refused a tenth. This is a report, not a permission to invent a colab ceiling
   number — **the ceiling is the owner's to set** (FOR THE OWNER 2). What you
   *can* do without a number: make `charge()` mark and print a per-job overrun
   whenever billed hours exceed the declared `est_hours` by more than a stated
   margin, on **every** backend. Both colab attempts declared 0.7 and billed 1.03
   and 1.08; neither left a mark.

6. **A PROBE's hours are invisible to `gpu_hours_no_verdict`, which is the one
   counter built to catch exactly this.** It joins charged jobs against
   `ledger.results` (`run.py:1456`), and a probe is *defined* as buying no ledger
   row. The T1.08 probe spent 2.6716 GPU-h against a ruling that AUTHORISED 1.20
   and `run status` still prints `T1.08: 0.36 h / 1 attempt / 1 verdict`. Extend
   the join to read `gpu_submissions.jsonl`'s `spec_phase` so probe hours land in
   a named `PROBE` bucket rather than vanishing. **Reporting-only and unfloored**
   per `D27`'s reasoning — probe spend is legitimate and gating it would punish
   the honest thing. Counting it is the point; a number nobody can see is the
   defect.

7. **Nothing in items 2–6 is a threshold and nothing in them is science.** If any
   of them turns out to need a bar, a ceiling or a spec, it is a routing, not a
   default — say so and route it.

---

## FOR THE OWNER

**1. DECISION NOT NEEDED, ACTION NEEDED — your loop was asleep for 26 hours and
the reason is other work on your account.** The builder's last productive slot
ended 2026-09-14 11:14 UTC. Nineteen consecutive hourly slots since then were
pace-skipped, and on the clock formula it releases around 13:00–14:00 today —
a ~26-hour blackout. The pacing arithmetic is correct and the 90% stop is your
own rule, so nothing here is malfunctioning. But `usage_attribution` reads: of
this week's 26 shared points, **builder 7 (26%), oversight desks 2 (7%), NOT
THIS PROJECT 17 (65%)**. This project is being paced out of its own meter by
work that is not this project, and the builder is the only organ throttled by it
(deliberately — the oversight organs keep the plain 90% gate so drift-catching
never gets starved). **You do not need to decide anything for the loop to
recover.** You should know that the cost has stopped being theoretical: one
armed deadline slipped ~14 hours, one paid-for GPU result went unread ~27 hours,
and `demonstrated` has been flat for 26.

**2. `D19` fired late, and I am telling you rather than extending it.** Its
`decide_by` was 2026-09-14; its NO-FETCH default was firable from 00:00 today and
did not fire, because the organ that fires defaults was pace-gated. I have not
fired it — that is the builder's act with pre-registered wording, and it is FOR
THE BUILDER item 2. **The deadline was not extended and must not be**; a deadline
that moves when it is reached is the deadlock it replaced. If you want to rule on
`D19` before it fires, it is options (i) capped `/data` cache, (ii) relocate
`HF_HOME`, (iii) decline — and it costs 3 specs (HR.1–HR.4, the hearing
programme's speech half).

**3. DECISION NEEDED — should colab have a weekly ceiling, and what is it?** I am
flagging this rather than appending it to `DECISIONS_NEEDED.md` as a new armed
row, because the pile is the current problem and this needs one number from you
rather than a fork. Today `remaining("colab")` returns **infinity**: colab hours
are billed to the week counter and constrain nothing, `afford()` always says yes,
and the overrun mark is hard-coded to kaggle. The T1.08 probe put 2.11 h through
that lane on two attempts that retrieved nothing, and no mechanism would have
refused a third, fourth or tenth. If colab genuinely is free and unlimited for
you, say so and the right repair is a comment stating it, not a ceiling. If it is
not, the builder needs the number. **The builder must not pick this number
itself** — that would be an organ setting its own budget.

**4. NO-DECISION — the queue is six promises overdue and the Review said so
first.** `review_queue_violations` moved 0 → 6 at midnight; all six were among
the thirteen the Review re-armed in the open on 09-14, after publishing a
stop-rule binding itself and the honest evidence against its own `D28` default.
Four consecutive days (09-15, 09-16, 09-17, 09-20) each carry 6 rows against a
measured capacity of 6. I am recording this as a desk that is behind and has said
so, not as a desk hiding anything. `D28` is due 09-21 and the Review has
recommended amending its own default before it fires; that recommendation is on
your desk and I agree with it.

**5. NO-DECISION — the four commitments you described most vividly have had
nothing runnable behind them for twelve days.** `claim_dead = 4, unchanged since
2026-09-03`: **smell**, **balance**, **thermal (too cold kills him)**, and
**shelter (he builds a shelter)**. Every park was legal and evidence-backed, and
no PARKED marker should be touched. But three of the four successors (`SH.02`,
`SM.03`, `BA.03`) are redesigns owed by the Review desk in item 4 above, and
`PARK-ON-AN-UNREACHABLE-RELEASE` stands at 3 — parks whose stated revival path
cannot be walked today. Nothing for you to rule on. It is the single most
goal-relevant number on this page and it has not moved in twelve days, so it gets
said out loud rather than left in a tool's output.

**6. NO-DECISION — the ledger is clean and I looked hard.** 108 PASS rows, every
one resolving to a live spec and a live commit save one dirty row an existing
instrument already classifies. Zero loosening in seven days; the only threshold
movements were three specs **tightened** with the arithmetic stated. `SO.10`'s
bakeoff disqualified its own headline winner on a pre-registered second gate and
left the seat vacant. Whatever else is wrong today, nothing on the scoreboard is
a lie.
