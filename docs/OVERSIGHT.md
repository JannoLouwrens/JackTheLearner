# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **100th audit — 2026-09-18, 18:37–19:0x UTC.** Opened at HEAD `a01837f`,
> fifteen minutes after the Review's held 18:22 sitting committed it.
> `demonstrated` **108/249 (43.4%)**, flat since 2026-09-14T04:18 — **4 d 14 h**,
> the fourth consecutive zero-net day.
> Builder: **29 commits in 24 h**, 6 iterations, **4 × rc=0**, 2 × rc=1 (both
> session-limit, 15:07 and 16:07, both recovered and the marker cleared at
> 17:14). Meters now: `week:all models` **36%**, `week:Fable` 50%, session 15%.

## VERDICT: DRIFTING

**The 99th audit's INTEGRITY RISK is CLOSED, and I checked it rather than
assuming it.** `docs/DECISIONS_NEEDED.md` now holds **28 `DECIDE:` blocks and 28
distinct ids** (`grep -o '^DECIDE: D[0-9]*' | sort | uniq -d` → empty);
the colab block was renumbered to `D31` at line 7030 with its heading, `D30`
was left on the blackout entry as instructed, neither was deleted, and
`decisions --check` exits **0** with the duplicate guard quiet. That was RANK 1
for 44 hours and it is done.

**The ledger is clean and I re-derived it independently.** 108 PASS / 26 FAIL /
14 VOID / 1 BLOCKED across 149 rows. All 108 PASS ids resolve in `BY_ID`; all
108 `commit` fields resolve in git; all 108 specs declare a `control`; 106 carry
`control_metrics` and the two that do not (`T0.01`, `T0.10`) declare
`control = "NONE, BY DECISION (52nd audit B5)"` with a written argument. Every
impl path exists. **Nothing on the scoreboard is a lie.**

So why DRIFTING. Because four things are simultaneously true and they point one
way:

1. The owner is holding a decision that **expires today** while the page they
   read tells them it expires **2026-09-25** (RANK 1).
2. **27.78 free Kaggle GPU-hours expire at the end of Saturday 2026-09-19** —
   ~29 hours — and `coverage` says **every GPU cost class is NOT FILLABLE**.
   Fourth such expiry (RANK 3).
3. Of 29 commits, **~21 served the instrument and ~5 served the ladder's
   science.** The instrument work was good and audit-ordered; it is still the
   machine tending the machine (§3).
4. `one brain / unison` — the constitutional core — has **27 specs and 1
   passing claim.** `curiosity`: 12 specs, 2 passing. Four commitments have no
   spec at all; eight are CLAIM-DEAD (§3).

Not INTEGRITY RISK: no threshold moved loose, no control weakened, no PASS is
unbacked, no ratchet was bought. DRIFTING is the accurate word and §8 says what
it costs.

---

## THE FOUR INSTRUMENTS

| instrument | exit | reading |
|---|---|---|
| `coverage` | **2** | 4 commitments with **NO declared spec** (`heavy`, `far`, `tiring`, `worth-it`), **8 CLAIM-DEAD of 29**, `NO-LIVE-PATH` 11 (upper) / 7 (lower). `unreachable = 97` AT floor. Unchanged from the 99th; the exit is the honest register the builder grew on 09-18, not a regression. |
| `decisions --check` | **0** | **0** hard violations — down from 1. 0 `MEANS-ESCALATED`, 0 `UNDECLARED`, 0 `OVERDUE`, 0 unrouted/vanished owner-asks. 1 `DEFAULT-ACTION-SAME-DAY` (`D30`), 5 soft `CONDUCT-MISFILED?` (`D20`, `D27`, `D28`, `D29`, `D31`). **`ratchet ok` printed in full** — the 99th's display-loss note is moot at exit 0. |
| `champions --check` | 0 | 10 violations, **every class AT its declared count and none moved**: 0/0 phantom arena, 2/3 unfalsifiable, 2+1/4 uncontestable, 4/4 unwinnable, 2/2 unverified verdicts, 3/3 trigger debt, 1/1 kindless. No new `ARENA-MISSING`, no new `NO-ARENA`. |
| `run review-queue` | **2** | **10 OVERDUE** (was 11 at the 99th; the −1 was the builder repairing its own MALFORMED stamp, not a disposal). 47 live, oldest live 25 d, **drain UNBOUNDED**, arrivals 16 vs disposals 7 over 7 cycles. **18 dated rows fall due on or before 2026-09-19 against a measured capacity of 6.** |

**Nothing to arm, and I am saying so rather than manufacturing an entry.**
0 `UNDECLARED`: every open decision carries a class, and all six `goal`-class
entries carry a `default` and a `decide_by`. 0 `MEANS-ESCALATED`. **0 defaults
are due to fire today** — `D20` and `D30` are due *on* 2026-09-18, which has not
ended, so firing either now would be taking a day the owner still has. See
RANK 2 for what must happen at 00:37.

---

## RANK 1 — The owner is being told a decision expiring TODAY expires on 2026-09-25, on a page written 15 minutes ago, *after* the repair that was supposed to fix exactly this

The 99th audit found the `D30` duplicate and named its owner-facing cost: the
Review's page quoted the colab entry's `decide_by` beside the blackout entry's
text. **The renumber fixed the tool. Nobody re-read the page.**

`docs/PROGRESS.md` `FOR THE OWNER`, written at 18:22 today (`a01837f`), line 177:

> **1. `D30` — cited, not re-asked (`decide_by` 2026-09-25), with one new
> fact.** … Nothing new is asked of you here.

The register, unchanged and correct:

```
$ sed -n '6938p' docs/DECISIONS_NEEDED.md
  decide_by: 2026-09-18
$ $PY -m experiments.decisions --check | grep -E '^\s+D30'
    D30    costs   0 specs   due 2026-09-18   [SAME-DAY RACE: must fire before its own 2026-09-19 event]
```

**The page contradicts itself in the same paragraph**, which is how you can tell
it is a carried-forward sentence and not a judgement: it states `decide_by`
2026-09-25 and then says *"`D30`'s own armed default fires on 2026-09-19"*. A
default cannot fire six days before its deadline. The 09-19 half is right; the
09-25 half is the ghost of the duplicate.

**What it costs, concretely.** `D30` is the pacing decision. Its own default
text says the perishable hours *"are NOT recoverable by any ruling made after
Sat 2026-09-19"*. The owner has **~29 hours** to rule on a question worth 27.78
GPU-hours this week and a recurring blackout, and has been handed **seven days**
and the words *"nothing new is asked of you here."* `PROGRESS.md` is
current-state by design, so it will not self-correct until the Review next sits
— at 06:37 on 09-19, after the deadline.

**And this is the part that decides B1.** The lesson that would have caught it
was written **five hours earlier, by this organ, about this exact defect** —
`docs/LESSONS.md:15371`, *"A DETECTOR SHIPPED WITHOUT ITS REPAIR CAN SILENTLY
REASSIGN WHICH DUPLICATE IS AUTHORITATIVE"* (99th audit, 12:37–13:0x today),
whose point 3 reads:

> **Fixing a register does not fix what was published from it while it was
> broken; go and look at the copies.**

The register was fixed at ~13:0x. The next thing published from it, at 18:22,
repeated the error. A written lesson did not survive five hours and one organ
boundary. That is the argument for an instrument rather than a fourth restating:
**the check has to be in `run status`, where a sitting cannot finish without
reading it.**

I cannot edit that page (it is the Review's, and not on my MAY list). The
instrument repair that stops this class recurring is **B1**.

A second, smaller instance of the same root, for completeness: the Review's
`FOR THE BUILDER` item 1 tells the firing slot to *"check the hour, not the
date."* `D30`'s `decide_by` carries **no hour** — it is the bare date
`2026-09-18`. The instruction is derived from `decisions.py`'s SAME-DAY-RACE
annotation rather than from the entry, and a slot that goes looking for an hour
in the entry will not find one.

---

## RANK 2 — Two armed defaults expire at midnight tonight. The 00:37 slot is the firing slot, and it is the only one before the thing `D30` is about becomes unrecoverable

Both `D20` and `D30` carry `decide_by: 2026-09-18`. Neither is OVERDUE yet —
correctly, the tool does not print them as such, and I have not fired them.
**At 00:37 on 2026-09-19 both read OVERDUE and both must fire that slot.**

Usage is healthy (`week:all models` 36%, resets 09-23 12:00), so the slot should
not be gated out — but note the precedent: five consecutive overseer slots were
`STOPPED at 91–100%` across 09-17/09-18, and `D19`'s default sat OVERDUE and
unfired through one of them. A default that can only fire in a slot that can be
paced out is not armed, it is hopeful.

**What each firing owes, written now so the slot does not have to re-derive it:**

- **`D20` → default (i) WALL STANDS.** The 57600 s detached wall is not raised,
  not narrowed, not re-based; `launch_detached.sh` is untouched; the builder
  registers no spec in `cpu<48h` while it stands. Journal with the words *"the
  owner did not rule by 2026-09-18, so the pre-registered default fired."*
  Reversal: the owner rules (ii) and the class re-opens. **No code change
  fires this** — the default *is* the standing posture, so firing it is a
  record, not an act. `cpu<48h` is EMPTY today, so nothing goes claim-dead.
- **`D30` → default (v) REPORT THE STREAK, GATE NOTHING, RELAX NOTHING.**
  `PACE_FLOOR`, `PACE_CAP`, the pace line and the 90% hard stop are all left
  exactly as they are. What ships is a standing `FOR THE OWNER` finding on the
  Review's page when a dark streak passes 2× the builder's cadence, printed
  beside the week's GPU-expiry forecast. Reversal: delete one paragraph from
  the Review's prompt. **Its own text says it fixes nothing and cannot save
  this week's quota** — fire it anyway and say that, because a default that is
  quietly skipped because it is weak is the deadlock defaults replaced.

---

## RANK 3 — 27.78 free GPU-hours expire in ~29 hours, the fourth week running, and the reason is not idleness — it is that every GPU door is shut behind a design ruling the Review desk has not had time to write

```
$ $PY -c "from experiments.gpu import Budget; print(Budget().remaining('kaggle'))"
27.7837
```

`Budget._week()` is `%Y-W%U` (Sunday-start, matching Kaggle's real reset), so
`2026-W37` closes at the end of **Saturday 2026-09-19**. W37 has charged 1.379 h
productive + 0.8373 h failed = **2.2163 of 30**.

**The builder is not at fault and I want that on the record.** Its last three
slot summaries each say, unprompted, *"no legal GPU dispatch remains on the
board, so W37's expiring hours are not a licence to manufacture one."* That is
exactly right and it is the hard version of honest.

`coverage`'s queue-depth block says why, mechanically:

```
  gpu<20min   0  EMPTY  <- NOT FILLABLE: pilot BLOCKED on evidence (DP.04, SM.03); the repair is a REDESIGN
  gpu<2h      1  UB.10  (no FRESH dispatch)  <- NOT FILLABLE: pilot BLOCKED on evidence (T2.11); the repair is a REDESIGN
  gpu<8h      0  EMPTY  <- NOT FILLABLE: pilot BLOCKED on evidence (LC.07); the repair is a REDESIGN
```

Every one of those four blockers — `DP.04`, `SM.03`, `T2.11`, `LC.07` — is a
**REDESIGN routed to the Review**, sitting as a queue row: `dp04-lifespan-has-
no-resolution` (DUE 09-22), `sm03-heldout-split-saturated` (DUE 09-22),
`t211-diayn-metric-cannot-separate-mi-from-noise` (**OVERDUE +2 d**),
`lc07-checkpoint-branch` (ACTED → `D24`, closed).

**So the chain is: unbounded queue drain → no design ruling → no pilot → no
legal GPU dispatch → free hours expire.** `D28` (the Review's ordering) and
`D30` (the pacing) are two ends of one rope, and neither is on my desk. Prior
expiries: W32 8.82 h, W33 22.11 h, W37 27.78 h pending. `D30`'s table already
counts this as three; it is four by Saturday night.

---

## RANK 4 — A disposition re-parented four of GOAL.md's own citations to a decision that had closed four days earlier. Nothing owns them now, and no instrument can see it

`coverage` prints, every run:

```
  4 NEW unrunnable citation(s) — fix GOAL.md's text or route the revival;
  never add to GOAL_UNRUNNABLE_BASELINE (shrink-only): GEN.02, GEN.03, GEN.06, GEN.09
```

These are `GOAL.md:193` and `:195` — *"MORE WORLDS … (GENERALITY.md GEN.06)"* and
*"OTHER MINDS … (GEN.02, GEN.03, GEN.09)"*. All four resolve, and all four are
welded behind `LC.07`, which is PILOT-BLOCKED. The queue row
`goal-cites-four-specs-that-resolve-to-corpses` was **ACTED 2026-09-16** on
commit `34116ca`, and its disposition reads:

> **Group B — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`** (welded behind `LC.07`,
> arena declared VENUE-UNAFFORDABLE 2026-09-06, **live on the owner's `D24`**)…
> Group B is **re-parented to `D24`'s resolution**, not to a date. **Whoever
> closes `D24` inherits these four.**

`D24` was not live on 09-16:

```
$ grep -n 'D24' docs/DECISIONS_RESOLVED.md | head -1
939:## D24 — RESOLVED BY ARMED DEFAULT (fired 2026-09-12 ~17:3x UTC, builder): (iii) DECLARE, DO NOT DECIDE…
```

**It closed on 2026-09-12 — four days before the disposition that re-parented to
it.** The disposition even quotes `D24`'s outcome (*"arena declared
VENUE-UNAFFORDABLE"*) without noticing that the decision which declared it had
already fired. And the ruling was (iii) DECLARE, DO NOT DECIDE — `LC.07` stays
PILOT-BLOCKED permanently — so *"whoever closes `D24`"* is **nobody, ever**.

**Scope it honestly.** No number is wrong. `goal_unrunnable` is truthfully 7,
`GOAL_UNRUNNABLE_BASELINE` was correctly not widened, the red was deliberately
kept rather than cleared by deleting citations, and the desk explicitly refused
the tempting exit. The reasoning in that row is *good*. The defect is one
factual premise, and its consequence is that four of the constitution's own
citations are owned by a closed decision. This is the `HOLD-ON-A-RESOLVED-
BLOCKER` shape — but the row is `ACTED`, and `review_queue.py` only checks
blockers on `HELD` rows, so **no instrument in this repo will ever look at it
again.** Repair routed as **B2**.

---

## RANK 5 — The Review's stated remedy for its own backlog is arithmetically short by its own instrument, and the desk should know that before Sunday

`docs/PROGRESS.md` `FOR THE OWNER` item 7 commits to a remedy and asks to be
checked against it:

> *Sunday's FULL is where I attempt the remedy … dispose the overdue class first
> and in bulk, ruling rather than re-dating, before Part 2 opens a single
> certificate.*

**The mechanism is sound and I am correcting a stale claim in `D27` while I am
here.** `D27`'s text says the Review *"has died at max turns on four of four
Sunday FULLs"* — that was true before 2026-08-31, when the builder raised
`review.sh` to derive 40 m / 120 turns. Since: **08-31 FULL completed**, **09-06
FULL completed and disposed six dated rows**, **09-13 FULL completed and opened
Part 2**. Three for three. The slot works.

**The arithmetic is the problem, not the slot.** `review-queue` measures the
consumer's best-ever cycle at **6 dated rows**. Today there are **10 OVERDUE**,
and **18 live dated rows fall due on or before 2026-09-19**, of which the tool
says **12 cannot be discharged by that cycle**. The 09-20 column alone carries
6. A remedy that clears at most 6 against a class of 10-and-growing does not
bend a drain that is running +9 over seven cycles — it holds it steady at best.

This is a measurement, not an accusation; the instrument says explicitly that a
slow week is legal and that this is never a violation. I report it because the
desk asked to be checked and because **24 of the ladder's 26 owned FAILs are
owned by nothing but a dated promise from that desk** (`fail_unowned` is
honestly AT floor 0; `FAIL-OWNED-BUT-UNDRAINED` is 24). If the Sunday attempt
does not bend the drain, the desk's own escalation — *"whether the Review's
cadence or its jurisdiction is wrong"* — is the owner's, and it should arrive
with this number attached.

---

## THE AUDIT

### 1. Integrity of the ledger — NO FINDINGS

Re-derived independently against `BY_ID` and git, not taken from the 99th:

| check | result |
|---|---|
| PASS rows | 108 |
| PASS ids missing from `BY_ID` | **0** |
| PASS `commit` fields unresolvable in git | **0** |
| PASS specs declaring no `control` | **0** |
| PASS rows with empty `control_metrics` | **2** — `T0.01`, `T0.10`, both declaring `control = "NONE, BY DECISION (52nd audit B5)"` with a written argument |
| PASS impl paths missing on disk | **0** |

`run status` adds, and I read them as legal and correctly reported rather than
as findings: **3 UNBACKED CERTIFICATES** (`LF.02` → `T6.03` → root `T2.10` FAIL;
`T2.03` and `T2.14` → `T1.08`) — the runs happened, nothing is invalidated, but
they cannot be re-bought until the dependency is; and **`T0.27`
deliberately-red** at `live_violations = 3`.

### 2. Thresholds and controls, over time — NO FINDINGS, and this is a real result

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`. Every numeric constant
that moved, moved **tighter**, and each carries its arithmetic:

- `MIN_DISTRACTOR_EVAL` **30 → 59** in `me_1`, `me_3`, `me_5` (γ=0.05 minimum).
- `MIN_DISTRACTOR_EVAL` **9 → 12** in `me_9` (`4f62755`, 79th audit item 6 —
  80% of a 15-pair censored pool; the family's 30 is unsatisfiable there, and
  the commit says so).
- `T1.07` `hypothesis`/`falsified_by` **STRENGTHENED** (`445b9e1`), header
  comment *"Strengthen-only: the old text is unchanged and a conjunct is
  ADDED"*, `MAX_SPREAD_RATIO 6.0` pre-registered with its reachability, and the
  certificate **deliberately staled and re-bought today** at attempt 5.
- `T1.08` `falsified_by` gains the 7% seed-CV conjunct; `MAX_HELDOUT_CV_PCT 7.0`
  explicitly does not move, on any probe branch.
- `N_PROPERTIES` **15 → 16 → 17 → 18** on the instrument batteries.

The `or`s added to `falsified_by` are **additional ways to be falsified**, which
is tightening, not the loosening `_check`-gains-an-`or` pattern. `seeds=3`
everywhere it appears; no seed count reduced; no control deleted or weakened;
`ME.9`/`ME.10` explicitly record *"THE 0.95 BAR DOES NOT MOVE in either
direction"* while adding the honest certified-vs-observed reading. **No silent
loosening.**

### 3. Drift from the goal

**What the builder did (29 commits, 24 h), by what it serves:**

| work | GOAL.md sentence |
|---|---|
| `HR.1` implemented + run (`5283aad`, `467cf1b`) | *"EVERY SENSE A HUMAN HAS"* — hearing; and *"Really learning, not appearing to learn"*. **The one unambiguous ladder act of the day.** |
| `T1.07` re-buy PASS attempt 5 (`22c5c9b`) | *"every capability claimed only by an experiment that could have failed"* — a certificate kept current under a strengthened claim. |
| `T1.08` §9d probe + BOTH_ABOVE ruling (`425a7e3`, `583a1e9`) | serves the ladder indirectly: `T1.08` blocks 45 specs. |
| B4/B5/B6, review retry lane, coverage register, 4 certificate re-buys (~21 commits) | *"protects the honesty of watching what happens"* — legitimate under GOAL.md's first principle, and audit-ordered rather than self-discovered. |
| journal ×3, lesson ×1 | conduct. |

**Nothing is drift in the strict sense** — every commit traces to a GOAL.md
sentence, and I looked for one that did not. But ~21 of 29 serve the fourth
clause (protecting honesty) and ~5 serve the first three (brain, body, world).
The Review named this itself and called it *gravity* rather than laziness. I
agree with the diagnosis and add the number: **this is the seventh consecutive
day at that ratio.**

**The converse, which is the harder question.** What has NO passing spec:

- **NO SPEC AT ALL (4):** `heavy`, `far`, `tiring`, `worth-it` —
  `GOAL.md:187`'s survival primitives. Registered as commitments on 09-18; the
  fork (write claims, or correct the sentence) is queue row
  `goal-187-names-seven-primitives-four-have-no-commitment`, **DUE today,
  unruled**, and the Review carries it to the owner as item 6.
- **CLAIM-DEAD (8):** `smell`, `balance`, `thermal (kills)`, `shelter/building`
  + the four above. Every claim spec PARKED or FORECLOSED. Three park→release
  pairs whose revival path cannot be walked.
- **LIVE CLAIMS, NOTHING PASSING (9):** `touch`, `tool use`, `told world`,
  `proprioception`, `plasticity`, `sleep`, `hunger/thirst`, `death & retry`,
  `fast/slow`.
- **The headline, and it is the one the owner should feel:**
  `one brain / unison` — **27 specs, 1 pass.** `curiosity` — **12 specs, 2
  pass.** These are the two claims GOAL.md is actually about. `hearing` is 14
  specs / 1 pass. `social/other agents` (14/4) and `language (parent)` (15/3)
  are the healthiest families on the board, and neither is the thesis.

### 4. Is the builder alive and productive? — ALIVE, and it had its best day in ten

6 iterations in 24 h: 13:20 rc=0, 14:17 rc=0, **15:07 rc=1**, **16:07 rc=1**,
17:14 rc=0, 18:25 rc=0. Both rc=1 are honest session-limit marks (`SESSION LIMIT
on every model — marking the lost iteration`); the 17:07 slot inherited both and
`recovered — clearing 2 lost-iteration marker(s)`. PASS delta **107 → 108 → 108**
(net 0 for the day; the `+1` at 14:17 is `T1.07`'s re-buy crossing back).

**All 8 of the 99th audit's `FOR THE BUILDER` items were executed within six
hours, and I verified each on disk rather than from the log summary:**

| # | item | evidence |
|---|---|---|
| 1 | renumber `D30`→`D31` | `56954e0`; 28 blocks, 28 distinct ids, 0 duplicates |
| 2 | Review retry path | `51c61d4` + `923661e`; **held live at 18:22 on the third poll** |
| 3 | prune `declared_pids` (third asking) | `/data/jack-logs/declared_pids`: both `run_spec HR.1` rows `EXITED`-stamped; only the live `review.sh` and `overseer.sh` slots undeclared |
| 4 | per-job GPU overrun mark, all backends | `2bfa84f`; `T0.12` re-bought clean |
| 5 | PROBE hours visible | `c9b0608`; live reading names `PROBE 3.59 h / 4 jobs`, `PILOT 2.00 h / 2 jobs`; `T0.36` re-bought clean |
| 6 | pace-skip notices detached artifacts | `cb38947`; fixture-verified; `T0.33` re-bought clean |
| 7 | `T1.07` re-buy | `22c5c9b` PASS, reattach not re-dispatch, charge idempotent |
| 8 | do not touch the uncovered commitments' ladder half | respected — the register grew, no spec was invented |

Two errors were self-reported and repaired on the record the same slot: `HR.1`
attempt 1 stamped `+dirty` (re-run clean, identical numbers), and the `t108`
queue row's 5-field MALFORMED stamp (`review_queue_violations` 11 → 10, text
preserved verbatim). **Self-reported errors are worth more than clean days**;
this is the behaviour the ledger depends on.

### 5. Compute honesty

- **Kaggle W37: 2.2163 h charged of 30. 27.78 h expire end of Saturday 09-19.**
  See RANK 3.
- **Colab W37: 3.033 h, no ceiling** — `remaining('colab')` is `inf`. `D31` is
  the entry; its default (i) **shipped today** (`2bfa84f`, observe-only). See §6.
- **`gpu_hours_no_verdict` = 48.42 h TOTAL.** The two new buckets are the honest
  news: `PROBE 3.59 h / 4 jobs` and `PILOT 2.00 h / 2 jobs` previously read as
  zero waste. The dominant line is unchanged and unowned by any new work:
  **`D1.0` 33.78 h across 2 attempts, 0 verdicts** — 70% of the total, and
  `D1.0` is a VACANT champion seat whose arena resolves to a `VOID`.
  `UNATTRIBUTED` 6.32 h / 21 jobs, `gpu_unattributed_jobs` AT floor 21.
- `overruns` is `[]` and will stay so until the next dispatch — the mark shipped
  after the two 09-14 colab overruns it was built from. Not a defect; a note so
  the empty list is not read as "no overruns ever happened."

### 6. Stuck decisions

**Nothing is escalated that a measurement could settle** (0 `MEANS-ESCALATED`).
**Nothing is unarmed** (0 `UNDECLARED`). Six open: `D20` (due today), `D27`
(09-20), `D28` (09-21), `D29` (09-22), `D30` (due today), `D31` (09-25).

**Was any owner-decision quietly acted on without being recorded? — One, and it
was recorded, but not where the owner will look.** `D31`'s armed default is
**(i) MARK BUT DO NOT CAP**, `decide_by` 2026-09-25. That option **shipped in
full today** as the 99th audit's B4 (`2bfa84f`): `charge()` marks and prints a
per-job overrun on every backend against the dispatcher's own declared
`est_hours` at a stated 25% margin; no ceiling invented, `remaining('colab')`
untouched, no dispatch refused. The builder's commit says *"the ceiling question
is `D31`'s (decide_by 09-25) and stays the owner's"* and the Review told the
owner on `PROGRESS.md` item 3. **Nothing was smuggled.** But the `D31` entry
itself still presents (i) as an unexecuted option, so an owner reading the
entry cold does not know the remaining question is only (ii) vs (iii). I have
appended an evidence-only addendum saying so; I did not resolve it, narrow the
options, or move the date.

**Five of six open entries carry the soft `CONDUCT-MISFILED?` flag** (`D20`,
`D27`, `D28`, `D29`, `D31` — class `goal` but blocking no spec id). I looked at
whether any is the desk's to take rather than the owner's, because that is the
`D1` disease. My reading: **`D28` is the one worth the owner's attention on this
point** — it asks permission for the Review to order its own work, the Review
has already announced it will attempt exactly that on Sunday, and an organ
ordering its own sitting is conduct by any reading. The other four each turn on
a number or an authority that is genuinely the owner's (a wall ceiling, a
budget, a seat marking, a GPU cap). I am flagging, not reclassing: `SYSTEM.md`
puts the class on the author, and four of these five are the Review's entries,
not mine.

### 7. Bakeoff hygiene

`docs/DECISIONS_RESOLVED.md`: no decision made without a learning gate in the
window, no winner chosen inside a noise margin, no new VOID-as-verdict.

The one standing instance is old, known, and **correctly still red**: `D10`
seated `wm-latent` **BY VERDICT off `LC.03`, which is a VOID**. `champions
--check` prints it as `VERDICT-IS-A-VOID` + `TRIGGER-UNREACHABLE` every run,
`unverified_verdicts` is 2/2 AT floor, and the marking carries a single-arm
caveat on its face. That is the system correctly refusing to let itself forget.

`D24`'s firing (2026-09-12, default (iii)) was verified at firing rather than
asserted — `champions --check` exits 0 before and after, every ratchet counter
identical across the edit — and it deliberately declined to write a `VENUE:`
field the grammar would silently ignore. That is good practice and I am naming
it. Its one live consequence is RANK 4: it closed, and a row four days later
re-parented work onto it as though it were open.

### 8. The honest summary — are we closer to a curious humanoid that climbs the ladder?

**Marginally yes, and for one specific reason that is worth more than the flat
count.**

`HR.1` is the day, and it is the kind of day this project is for. A spec was
written to ask whether a corpus could honestly support the three speaker-ID
claims stacked behind it. Its author wrote down, *before running it*, which way
he expected it to fail. It failed that way: a 17-dimension probe reading only
the **silence floor** identifies enrolled LibriSpeech readers at 0.24 / 0.38 /
0.43 across seeds against a 0.10 bar and 0.05 chance, while the planted
same-session control reads 0.72–0.84 (instrument alive) and the noise stratum
sits at chance. **LibriVox readers keep their microphones across chapters.** Any
`HR.3` speaker number bought on that corpus would have been a measurement of
recording equipment. It cost ~16 s of CPU and it was caught before one GPU-hour
reached the family. `HR.2`–`HR.4` stay killed; the redesign is routed with three
arms priced.

That is a claim that could have been believed, refused before it was paid for.
It is the standard the whole ladder exists to enforce, and it held.

**And now the part that is not marginally yes.** `demonstrated` has been 108 for
four and a half days. The constitutional core — *one brain, all senses in
unison* — is 27 specs and **one** passing claim. Curiosity is 12 and **two**.
Four of the owner's own survival primitives have no falsifiable claim at all,
eight commitments are claim-dead, and the specs that would revive them are
redesigns queued behind a desk whose drain its own instrument calls UNBOUNDED.
Meanwhile 27.78 free GPU-hours expire tomorrow night for the fourth time,
because every door they could open needs a design ruling nobody has had a slot
to write.

So: **we are closer to a system that cannot lie to itself, and no closer to a
creature.** Those are not the same axis, and this project has spent seven days
moving hard along the first one. That was the right call while the register was
broken — it is fixed now. The next thing that should move is Jack.

---

## FOR THE BUILDER

Ordered by what stops being repairable first. Items 1 and 2 are instruments;
neither invents a threshold, refuses a run, or touches a spec.

1. **A steering page that quotes a `decide_by` must be checked against the
   register. RANK 1, and it is live right now.** `run status` already has a
   `STEERING-PAGE ORDERS` block that parses `docs/PROGRESS.md` for **spec ids**
   and resolves them against the registry. Extend the same parse to **decision
   ids**: for every `D<n>` on `docs/PROGRESS.md` and `docs/OVERSIGHT.md` that
   appears within the same sentence as a date, compare that date to
   `decisions.parse()`'s `decide_by` for that id and print
   `STEERING-DATE-MISMATCH: D30 — page says 2026-09-25, register says
   2026-09-18`. **Reporting-only and unfloored**, per `D27`'s reasoning: a page
   may legitimately cite a date for another reason, and a false positive here
   must not turn anything red. Known-positive for the fixture is on disk today:
   `docs/PROGRESS.md:177`. **Do not edit `PROGRESS.md`** — it is the Review's
   page and rewriting another organ's current-state file is not yours (or
   mine); the instrument's job is to make the next sitting unable to miss it.
2. **`review_queue.py`: an `ACTED` disposition that re-parents to a decision id
   must check that the decision is still open. RANK 4.** New reading —
   `DISPOSITION-ON-A-CLOSED-DECISION`, the `ACTED`-row analogue of the
   `HOLD-ON-A-RESOLVED-BLOCKER` you already implement for `HELD` rows. Scan
   disposition bodies for `D<n>` references in re-parenting position, resolve
   against `DECISIONS_RESOLVED.md`, and print the pair. **Reporting-only, never
   a violation and never floored** — re-parenting to a decision that later
   resolves is normal and correct; what is not visible today is re-parenting to
   one that *already had*. Known-positive on disk:
   `goal-cites-four-specs-that-resolve-to-corpses`, ACTED 2026-09-16, re-parents
   `GEN.02/03/06/09` to `D24`, which closed 2026-09-12. **Do not touch the row,
   the citations, or `GOAL_UNRUNNABLE_BASELINE`** — the disposition's refusal to
   clear its own red by deleting citations was correct, and the baseline is
   shrink-only in both directions. The row's repair is the Review's.
3. **The 00:37 slot on 2026-09-19 fires two defaults. See RANK 2 for what each
   owes.** If you are the slot that notices first: `D20` → (i) WALL STANDS
   (a record, no code), `D30` → (v) REPORT THE STREAK (one paragraph into the
   Review's prompt; `PACE_FLOOR`, `PACE_CAP`, the pace line and the 90% stop all
   untouched). Journal both with the words *"the owner did not rule by
   2026-09-18, so the pre-registered default fired"* and state the reversal.
   **Do not fire either before midnight** — today is still the owner's.
4. **W37's 27.78 GPU-hours: keep refusing to manufacture a dispatch.** Your last
   three slot summaries got this exactly right and I am confirming it rather
   than asking for anything. There is no legal candidate: every GPU class is
   pilot-blocked on a redesign owed by the Review. **Spending them on a spec
   whose gates are not frozen would be buying a number, and letting them expire
   honestly is the correct outcome.** It is also RANK 3, and the fix is not
   yours.

---

## FOR THE OWNER

**1. `D30` EXPIRES TODAY, 2026-09-18 — not 2026-09-25, and the page you read
says 09-25. This is the one item to read if you read one.**
`docs/PROGRESS.md` `FOR THE OWNER` item 1, written at 18:22 today, tells you
`decide_by` **2026-09-25** and adds *"nothing new is asked of you here."* The
register says **`decide_by: 2026-09-18`**. The wrong date is a sentence carried
forward from the duplicate-`D30` bug the builder fixed this morning — the tool
is correct now; the page was not re-read. The same paragraph contradicts itself
by stating the default fires 2026-09-19, which cannot be true of a 09-25
deadline.

**The question `D30` asks:** should the builder's pacing line read the shared
`week:all models` meter — 75% of which is consumed by things that are not this
project — or this project's own attributed spend? The desk's recommendation is
**(i) pace against our own attributed spend**, with the 90% all-models hard stop
**untouched**, because `pace_gate` is only consulted after `usage_gate` has
already said yes. The desk recommends *against its own default* and explains
why it will not take (i) itself: a default may not loosen a gate, and an organ
may not vote itself more budget.

**If you say nothing, default (v) fires at 00:37 tomorrow** — report the streak,
change no gate, fix nothing. Its own text is blunt that it cannot recover this
week's hours. **Reversal at any time: delete one paragraph from the Review's
prompt.**

**2. `D20` also expires today**, and its default (i) WALL STANDS is the status
quo — the 57600 s detached wall is not raised or narrowed, and no spec is
registered in `cpu<48h` while it holds. Firing it costs nothing and changes
nothing; it just stops the entry sitting open forever. Nothing is claim-dead
behind it.

**3. 27.78 free Kaggle GPU-hours expire at the end of Saturday 2026-09-19 —
about 29 hours — and there is no legal way to spend them.** Fourth week running
(W32 8.82 h, W33 22.11 h, W37 27.78 h). **The builder is not idling and refused
three times today to invent a dispatch**, which is the right answer. The cause
is structural: every GPU cost class is blocked on a spec whose gates cannot be
frozen until the Review writes a redesign, and the Review's queue drain is
UNBOUNDED at 47 live rows. `D28` and `D30` are the two ends of that rope. I am
reporting this, not asking — but if you want one number to judge the project's
throughput by this week, it is this one.

**4. NO-DECISION, reported because you are entitled to the correction: `D31`'s
own default has already shipped.** Option (i) MARK BUT DO NOT CAP — the per-job
overrun mark on every backend — landed today as `2bfa84f`, observe-only, no
ceiling invented, no dispatch refused. It was ordered by the 99th audit as
instrument work and the builder correctly flagged that the *ceiling* question
stays yours. So when `D31` reaches its 09-25 deadline, the live question is only
**(ii) give colab a ceiling** vs **(iii) decline** — (i) is done. I have appended
that as evidence to the entry and changed nothing else about it.

**5. NO-DECISION, liveness: every organ is alive and the one that was broken is
fixed.** Builder 6 iterations / 4 rc=0 / 29 commits, all 8 audit items
discharged in six hours. The Review's retry lane (`923661e`) worked on its first
live test and gave the desk back a day it had lost twice. Field watch on cadence
(next 09-21). Overseer on cadence. Usage 36% of the weekly meter.

**6. Carried forward so it does not vanish, and still NOT ripe for you:**
`GOAL.md:186-188` names seven survival primitives — *hot, heavy, far, tiring,
dangerous, worth-it, that-person-lied* — and four have no falsifiable claim. The
fork (write the claims, or correct the sentence) is the Review's queue row
`goal-187-names-seven-primitives-four-have-no-commitment`, **DUE today and
unruled**. The builder did the monotone half correctly: the four entered the
commitment register, `claim_dead` rose 4 → 8, which is a hole becoming countable
rather than a regression. The desk owes you its recommendation before this
becomes a `D` entry, and I am not pre-empting it.

**7. The honest bottom line.** This project spent the last week making itself
unable to lie, and it succeeded — the register is repaired, no threshold moved
loose, the ledger is clean on every check I can run, and today a brand-new spec
caught a corpus lying before it cost a GPU-hour. It has not moved Jack. 108
demonstrated for four and a half days; *one brain, all senses in unison* stands
at 27 specs and **one** passing claim. The instruments are now in good enough
shape that the next week should be spent on the creature.
