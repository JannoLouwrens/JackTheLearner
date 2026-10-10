# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-10 06:37 UTC — 148th audit.**

## VERDICT: INTEGRITY RISK

> **UPGRADED FROM `DRIFTING` AT 06:49, MID-AUDIT, AND THE REASON IS THE WHOLE
> REPORT.** I drafted this page with a `DRIFTING` verdict and a §1b that said
> *"no certificate lost its implementation this time."* **That became false at
> 06:45:06**, while I was re-running the instruments before committing. The
> 2-hourly regate cron fired during the Review's live sitting and re-bought
> **`T0.31`** against a `docs/REVIEW_QUEUE.md` the Review was mid-edit; the row
> now reads *"the code that ran was never committed, reconstructs from no commit
> and was not preserved … **it cannot be recovered by anyone**."* That is the
> 146th audit's exact sentence, two days later, on the spec that gates the
> backlog reader. The near-miss I was reporting is a hit. Full detail in §1b.

**The ledger's numbers are clean and one of its Tier-0 certificates is now
unrecoverable. The organ machinery that is supposed to protect it has a hole
shaped exactly like the runs that fall into it, and the page carrying the
already-written repair is the page that is stale.**

`scripts/lib_seal.sh`'s rescue sweep — the branch that commits a dying run's
work so it is not lost — is reachable **only when `docs/PROGRESS.md` is itself
dirty**, and `scripts/review_prompt.md:154` orders that page written **LAST**.
So the sweep can fire only for a run that got far enough not to need rescuing.
Measured consequence: the 2026-10-09 Review wrote seven declines, `D43`, and
`THE CLASS DECLINE`, died `rc=124`, and its work sat **uncommitted for 24
hours** until today's run inherited it at `81480de` (06:38:10). In that window
the regate sweep re-bought three Tier-0 certificates **11–13 consecutive times
against a dirty tree**. They are the three that gate my own instruments.

Second, and the reason nobody saw it: **the seal's own stamp resets the clock
the seal reads.** On 2026-10-08 it printed *"docs/PROGRESS.md untouched by this
rc=124 run and only 24h old (cadence allows 25h) — still current, not
stamping"* about a page whose content was **96 h old and whose own first line
already said STALE** — because the 24 h was measured from the seal's own
2026-10-07 STALE stamp.

And the perishable loss the 147th audit computed one day early is now **fact,
not forecast**: `2026-W40` has **28.03 of 30 free Kaggle GPU-hours undrawn and
they expire today**. The builder — the only organ that can spend them — has
been **dark 41 consecutive slots / 41.6 h**.

---

## 1. Integrity of the ledger — CLEAN

**107 PASS rows of 160 results; 107/257 demonstrated.** Independently
re-derived, not taken from `run status`:

- Every PASS names a `commit`; **0** rows with a missing commit field.
- **0** PASS rows whose commit has vanished from git. Three carry a `+dirty`
  suffix (`T0.21`, `T0.28`, `T0.31` at `f24ac94+dirty`) — see §1b.
- **Only `T0.01` and `T0.10`** have empty `control_metrics`, both harness smoke
  rows that declare no control, both already owned by
  `t018-explicit-no-control-reads-as-an-unrun-promise` (OPEN, `DUE: 2026-10-19`).
  Every other PASS that declares a control has one recorded. **0 exceptions.**

### 1b. The dirty stamps — and `T0.31` is now unrecoverable

**At 06:39, when I first read it, this section was a near-miss.** `run status`
has a `! DIRTY STAMPS` reader, it ran, and it said of all three
(`T0.21`/`T0.28`/`T0.31` at `f24ac94+dirty`): *"the implementation it names IS
committed: `impl_sha` reconstructs byte-identically at `81480de3` … the
uncommitted edits were outside what this spec declares."* The reader works and
deserves the credit.

**At 06:45:06 that stopped being true.** The regate cron fired *during* the
Review's live sitting — between its 06:43:14 and 06:49:17 commits, while
`docs/REVIEW_QUEUE.md` sat mid-edit — and re-bought `T0.31`. The row now reads:

> `T0.31` recorded PASS; ran from a modified tree at `6b50392`; **the code that
> ran was never committed, reconstructs from no commit and was not preserved**
> (1 uncommitted code file(s) at run time, INCLUDING this spec's own
> `docs/REVIEW_QUEUE.md`) — **it cannot be recovered by anyone.** Re-run it from
> a clean tree.

**`T0.31` is "The backlog reader cannot be quieted by tidying the backlog"** —
the ratchet spec whose P4/P5/P6 assert on the violation TOTAL, named in this
organ's own standing instructions as the thing that stops a "repair" from
lowering its own number. Its implementation is gone. The reason it is
irrecoverable rather than merely dirty: `REVIEW_QUEUE.md` is a genuine
`IMPL_DEPS` member of `T0.31`, the re-buy consumed the Review's **intermediate**
working-tree state of that file, and the Review then committed a *different*
state four minutes later. The bytes that were tested exist nowhere.

**This is not a regression of the repair that was supposed to prevent it.**
`cross-organ-doc-race-voids-certificates` was ACTED 2026-10-04 (`b4df9bb`); it
made the overseer's *prose* docs (`OVERSIGHT.md`, `LESSONS.md`) `PROSE_DOCS`-exempt
and deliberately left `REVIEW_QUEUE.md` unexempt, **correctly**, because `T0.31`
genuinely reads it. What that repair never addressed is the scheduling half:
**there is no interlock whatsoever between the 2-hourly regate cron and the
organ sittings.** The cron does not know the Review is editing; the Review does
not know a re-buy is due. At 06:37 every day they are guaranteed to overlap.
`T0.28` happens to have escaped — it bought clean at 06:44:20 (`6b50392`) in the
same window — which is the whole point: **which certificate dies is decided by
cron phase.**

The volume and cause, measured at 06:39 before the loss:

| spec | consecutive `+dirty` re-buys | last CLEAN buy |
|---|---|---|
| `T0.21` The GOAL.md coverage audit cannot be flattered by a word | **13** | 2026-10-08T12:43:16 |
| `T0.28` The escalation tool can be shown catching a deadlock | **11** | 2026-10-09T06:44:21 |
| `T0.31` The backlog reader cannot be quieted by tidying the backlog | **11** | 2026-10-09T06:45:04 |

Those three specs gate `coverage.py`, `decisions.py` and `review_queue.py` —
**the first, second and fourth instruments this audit is required to run.** The
re-buy message reads `HASH-SALT DIFFERENTIAL CLEAN`, which is true of the salt
differential and says nothing about the tree; a reader skimming the row sees the
word CLEAN. The 146th audit (2026-10-08) measured this same mechanism producing
a `T0.28` whose implementation **could not** be reconstructed. Two days later it
fires every two hours and the only thing standing between it and that outcome is
which files happened to be dirty.

## 2. Thresholds and controls over seven days — CLEAN

**Not one numeric bar moved in the loosening direction.** Ten commits touched
`registry*.py` or `experiments/tests/`. I diffed every changed line carrying a
number or comparator (482 lines) and ran down every `-` hit by hand:

- Every constant is an **addition** — `W1.01`, `W1.04`, `T4.04`, `T4.05`,
  `T6.01` newly implemented; `T1.08`'s `TAIL_FRAC`/`WARMUP`; `LG.14`'s imported
  bars (*"Bars imported, none moved"*).
- The single `-` line with a comparator —
  `- and c["inference_params_trained_frac"] < 0.9` in `T1.11` — is the diff
  artifact of **adding a third conjunct** (`m["shipped_callers"] >=
  SHIPPED_CALLER_MIN`). Both original conjuncts are byte-unmoved and the change
  **demoted `T1.11` to FAIL**. A strengthening, recorded as one.
- `MAX_HELDOUT_CV_PCT` 7.0 byte-unmoved across all three `T1.08` Step-2b
  attempts, verified at source.
- No control deleted, no `_check` gained an `or`, no seed count reduced.

There is no `-CONST = N` / `+CONST = M` pair anywhere in the window.

## 3. The seal's two defects — THE AUDIT'S PRINCIPAL FINDING

### 3a. The rescue sweep is unreachable by the runs that need it

`seal_output()` (`scripts/lib_seal.sh:305–325`):

```
if [ -z "$(git status --porcelain -- "$file" 2>/dev/null)" ]; then   # :313
    ... age check ...            -> "still current, not stamping" ; return 0
    ... stale_output ...                                           ; return 0
fi                                                                   # :324
# The run's whole dirty set, partitioned ...                         # :325
```

Both clean-branch paths `return 0` **before** line 325, where the dirty-set
sweep that commits the run's *other* outputs begins. So the sweep is gated on
the sealed file being dirty — and `scripts/review_prompt.md:154` says
*"PROGRESS.md last, as the receipt for commits that already exist."*

**A run that dies before its last checklist item leaves `PROGRESS.md` clean,
and therefore gets no sweep at all.** That is every `rc=124` run, which is most
of them. Confirmed in the log: 2026-10-09 06:57:09 printed *"docs/PROGRESS.md
already carries a stale banner — leaving it"* and swept nothing, while
`REVIEW_QUEUE.md` and `DECISIONS_NEEDED.md` held that sitting's entire output.

This is the 146th audit's lesson 1 — *"the failure mode of a per-file liveness
check is a false negative about every other output"* — but one layer deeper
than that lesson went. It recorded the **sentence** as wrong. The same per-file
predicate also controls the **sweep**, so the consequence is not cosmetic: the
work stays uncommitted, which dirties the regate, which is what bills the
certificates in §1b.

### 3b. The seal's own stamp resets the clock it reads

`_seal_file_age_hours()` (`:227–232`) is `git log -1 --format=%ct -- "$file"` —
the last commit touching the file. `stale_output()` **commits that same file**
to add its banner. So every stamp resets the freshness clock to zero.

Traced against `docs/PROGRESS.md`, whose real content is `b61515f`
(2026-10-04 07:10:31):

| date | seal said | content actually was |
|---|---|---|
| 10-05 06:57:13 | sealed as INCOMPLETE draft (`bd877b9`) | 23.8 h |
| 10-06 06:57:11 | *"only 23h old … still current, not stamping"* | **47.8 h** |
| 10-07 06:37:04 | stamped STALE, *"last moved 47h ago"* (`1b7a3ea`) | **71.4 h** |
| 10-08 07:08:52 | *"only 24h old … still current, not stamping"* | **96.0 h** |
| 10-09 06:57:09 | *"already carries a stale banner — leaving it"* | **119.8 h** |

On 10-06 and 10-08 the instrument measured its own prior commits and declared
the page current. **The 10-08 reading called a page "still current" whose own
first line reads "STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED
NOTHING."** Two artefacts in direct contradiction and no reader joining them.

Note the bias direction. The 146th audit's lesson named the *false negative*
(the seal under-reports what a run produced). **This is the false positive — the
seal over-reports freshness — and it is the direction that flatters.** A
double-stamp guard exists; a self-reset guard does not.

## 4. Is the builder alive and productive? — NO, AND LEGALLY SO

- **Last iteration that ran at all: 2026-10-08T12:32:55, `rc=0`.**
- Last 24 h: **40 hourly slots, 40 `PACE-SKIP`, 0 `rc=0`, 0 ledger events.**
- At 06:07 today: *"41 consecutive dark slot(s); 0 failed slots (41.6 h since
  the last rc=0)."*
- Cause is the pace line, not a crash and not the hard stop: `week:all models`
  **64 %** against a pace line of **51 %**, hard stop 90 %. **73 % of this
  week's 53 shared points belong to another tenant** (builder 8, desks 6).

Three declared dispatches (`T0.21`, `T0.28`, `T0.31`) have been reported EXITED
since 2026-10-08T12:32:55 with the notice *"the next unskipped iteration should
read them"* — repeated every hour for 41 slots to a builder that cannot wake.
Not a thrash and not a failure: a correctly-paced organ with nothing to pace.

## 5. Compute honesty — THE LOSS IS NOW REALISED

`experiments/gpu_budget.json`, `weeks` block:

- **`2026-W40`: 1.971 h drawn of 30 free Kaggle hours → 28.029 h expire TODAY
  (Saturday 2026-10-10).**
- W37 1.379 · W38 0.9176 · W39 1.0719 · W40 1.971 → **≈114.6 h undrawn across
  four consecutive weeks.**
- The 147th audit computed this yesterday and named the arithmetic: `pace_gate`
  cannot release the builder until the usage week is 64 % elapsed ≈ 2026-10-12
  20:30 UTC, **~2.4 days after the hours die.** Nothing has changed; the
  forecast is now the outcome. It is recorded here as realised, not re-forecast.
- **No GPU hours were spent without a ledger entry** — every W40 job
  (`1791217029`, `1791444041`, `1791461828`) has a charged row and an `ok: true`.
- **Two per-job overruns, both in W40, both under-estimates:** est 0.5 h →
  **0.6544** billed (+31 %), est 0.5 h → **1.0458** billed (+109 %). Honestly
  recorded in `overruns`. Flagged because `T1.08` Step 1's projection is 0.3 h
  and the only two recent calibration points both ran long.

## 6. Stuck decisions — ARMED, AND ONE FALLS TODAY

`decisions --check` **EXIT 0**, ratchet ok — `0/0` undeclared, `0/0`
unrouted-owner-ask, `0/0` vanished-owner-ask, `0/0` default-action-expired,
`0/0` firing-diff. 31 of 31 firings transcribed.

- **No `MEANS-ESCALATED`.** No measurable fork is sitting on the owner's desk —
  the `D1` disease is absent.
- **No `UNDECLARED`** to arm, so this audit arms nothing: the register is
  already fully armed. **No `OVERDUE — DEFAULT IS DUE TO FIRE`:** `D43`
  2026-10-16, `D41`/`D42` 2026-10-18.
- **`D40`'s `decide_by` is TODAY, 2026-10-10** (default (v) = the status quo).
  It is the 90 % hard-stop item and it is the decision whose subject just cost
  the project 28 GPU-hours. Flagged, not fired — its date has not passed.
- Seven conduct entries are unarmed by design. **`D33` is STALE by 17 days**,
  `D35` by 16, `D38` by 6. `D33`'s default is **MOOT, not merely expired** — its
  object went terminal when `w1-world-edit-window` was DECLINED, so no desk can
  clear it by firing anything. That reading is the 10-02 Review's and I
  re-derived it rather than inheriting it: the entry's recommendation moves W1
  design authority to the builder, and `D22` (the owner's resolved ruling) puts
  it with the Review. The condition cannot age out.
- `D41`/`D42` carry `CONDUCT-MISFILED?` — class `goal` but blocking no spec id.
  Soft, routing-only, correctly left to the Review.

**`D43` was opened by the 10-09 Review and is honest about its own cost** — it
states that seven findings now sit in TERMINAL rows, that a terminal row is
never re-read, and that the 10-11 split is load-bearing as a result. I checked
the one thing a desk could have got wrong here and it did not: the decline
executed a **pre-registered** stop-rule on its own date and its own condition,
and the findings are preserved rather than withdrawn.

## 7. Bakeoff hygiene — CLEAN

33 resolved entries in `docs/DECISIONS_RESOLVED.md`. No decision made without a
learning gate; no winner chosen inside the noise margin; every `screen
rationale` I read is **pre-declared before any arm number existed**, with the
gate *"unmoved at 3 sigma and MIN_FINISHERS still applies."*

One standing debt, correctly reported rather than hidden: the **Learning core**
seat is held `BY VERDICT` off `LC.03`, which is a **VOID** — a VOID decided
nothing. `champions --check` raises it as `VERDICT-IS-A-VOID` +
`TRIGGER-UNREACHABLE` and it sits at its floor. Not a new finding and not a
repair I may make; named so it is not mistaken for absence.

## 8. The instruments, re-run immediately before this file was committed

The Review is in flight in this same 06:37 slot (PID 2272208, and it committed
`81480de` while I was reading), so a reading quoted from the top of a sitting is
the default failure here.

Readings below are the **06:49** re-run, not the 06:39 one, and two of them moved
in those ten minutes:

| instrument | exit | reading (06:49) |
|---|---|---|
| `coverage` | **2** | **0** commitments with NO declared spec. **3 CLAIM-DEAD** (smell, shelter/building, thermal — every claim spec parked or foreclosed); 14 with live claim specs but nothing passing. |
| `decisions --check` | **0** | ratchet ok, all five classes at `0/0`. Now **4** armed — `D44` appeared at 06:49 (`due 2026-10-24`, `CONDUCT-MISFILED?`). |
| `champions --check` | **0** | ratchet ok; 10 violations, every class at floor. **`ARENA-MISSING` is 0.** |
| `run review-queue` | **0** *(was 2)* | **3 OVERDUE → 0.** The Review disposed all three during this audit: `hash-salt-lottery` ACTED (`5e3fb47`, 06:39:57), `lc03-five-controls` RE-DATED 2026-10-21 (`6b50392`, 06:43:14), `personality-is-a-typed-character-sheet` option (iii) with the character question routed as `D44` (`bd40dde`, 06:49:17). 45 OPEN, 48 ACTED of 139. **A clean board, honestly earned, and it is the one number that improved today.** |
| `run status` | **2** | 107/257; the `! DIRTY STAMPS` block of §1b — **now carrying `T0.31` as unrecoverable.** |

**A correction to my own standing instructions, because it cost me a wrong
reading mid-audit.** The overseer prompt says `champions.py` finds
*"`ARENA-MISSING` — **8 seats today**"*. It finds **zero**; the ratchet line
reads `0/0 seats with a phantom arena`. Every number in those instructions is a
dated example, not current state. (I also initially mis-read `coverage` as
EXIT 0 by piping it through `head` and reading `head`'s status. It is **2**. Do
not pipe an instrument whose exit code you intend to quote.)

## 9. Drift from the goal

**What the last day of work served.** The builder did nothing (§4), so the
day's output is the Review's: `T1.08` Step 2b stamped ACTED and its residual
routed as an estimator question; the seven-row class decline; `D43`. `T1.08` is
*"Seed variance measured"* — it serves **"Really learning, not appearing to
learn … at ≥3 seeds where the claim is about learning."** The decline and `D43`
serve *"protects the honesty of watching what happens when the three meet."*
**No drift: nothing was worked on that traces to no GOAL.md sentence.**

**The converse, which is the harder question.** `coverage` EXIT 2 names three of
the owner's own constitutional commitments with **no live falsifier at all**:

- **smell** — `SM.02` PARKED, `SM.03` FORECLOSED. GOAL.md: *"olfaction finds
  food, fire and decay at a distance and through occlusion — the sense that
  works when sight fails."*
- **shelter/building** — `SH.01` PARKED, `SH.02` FORECLOSED. *"the owner's own
  image of success."*
- **thermal (kills)** — *"too cold kills him, too hot kills him."*

All three are CLAIM-DEAD **compositionally**: each park was individually legal
and evidence-backed, and no dispatch anywhere revives the commitment behind
them. This is `D42` (armed, `decide_by 2026-10-18`, default (iv) HOLD) and
`five-commitments-are-claim-dead-behind-foreclosures` (OPEN, `DUE: 2026-10-14`).
It is correctly routed and correctly on the owner's desk; I am reporting that it
has stood for **eight days** and that the daily reporting of it has become
routine, which `D42`'s own price paragraph predicted.

Fourteen more commitments have live claim specs with nothing passing —
**touch, tool use, told world, proprioception, sleep, plasticity, fast/slow**
among them. Curiosity has 2 passing of 12; one brain / unison 1 of 28.

## 10. The honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday?
No. We are one day closer to not being able to find out.**

The honest accounting of the last 48 hours: the ledger did not move (107/257,
unchanged since 10-08), the builder ran zero iterations, 28 GPU-hours died, and
the entire output of both organs was **spent on repairing the organs**. That
work was good and necessary — `T1.08` Step 2b is a real negative result, the
class decline ended an eleven-day standing violation — but none of it is Jack
learning anything.

And the specific shape of the drift is worth stating because it is not laziness.
The system's self-repair machinery has begun to consume the capacity that would
otherwise go to the ladder, and its own instruments are now the thing most often
being fixed. **Three of the four instruments I am required to run are gated by
certificates that have been re-bought against a dirty tree 11–13 times in two
days, caused by a seal whose rescue path cannot reach the runs that need it, and
the repair for that was written down six days ago on a page that the same seal
has twice declared "still current".** That is a closed loop, and closed loops do
not open themselves.

The counterweight, stated because it is true: **section 2 is clean and the
backlog board is clean.** Nothing has been loosened, no PASS is unsupported, no
control went unrun, no GPU hour went unbilled, and the Review cleared every
OVERDUE row while I watched. The scoreboard is honest. It is just not moving,
and the reason it is not moving is no longer the science.

**One last thing, and it is the most uncomfortable sentence in this report.**
The certificate that died at 06:45 is the one that exists to stop an organ from
quieting a backlog by tidying it — and it died because an organ was tidying a
backlog. Nothing cheated: the Review's disposals were honest, the cron's re-buy
was honest, the stamp that recorded the loss was honest. **The system caught
itself destroying the evidence that it cannot destroy evidence, and the only
reason anyone knows is that a human-facing reader happened to print it.** No
exit code went red for `T0.31` specifically; `run status` exits 2 for a dozen
reasons and has for days.

---

## FOR THE BUILDER

**B0. THE REGATE CRON AND THE ORGAN SITTINGS NEED AN INTERLOCK, AND THIS IS NOW
THE TOP ITEM — it cost `T0.31`'s implementation at 06:45:06 today.** The
2-hourly regate (`scripts/regate.py`, no model call) re-buys specs whose
`IMPL_DEPS` include `docs/REVIEW_QUEUE.md` / `docs/DECISIONS_NEEDED.md`, with no
knowledge of whether an organ is mid-edit. Today it fired between the Review's
06:43:14 and 06:49:17 commits and recorded a PASS whose tested bytes were then
overwritten — *"it cannot be recovered by anyone."* At the 06:37 slot this
overlap is **guaranteed**, every day.
The repair is a **refusal, not a retry**: before buying a spec, if any file in
that spec's `IMPL_DEPS` is dirty, **skip that spec and log the skip** — a
re-buy is a convenience and a lost implementation is not recoverable, so the
asymmetry is total. `protocol.py` already computes exactly this predicate for
the `+dirty` stamp (`:220`, *"pulled out of the `+dirty` stamp so the question
can be asked of a fixture"*), so the check exists and only needs calling one
step earlier.
**Do NOT** fix this by adding `REVIEW_QUEUE.md` to `PROSE_DOCS` or to any
exclusion list. `T0.31` genuinely reads that file; exempting it would make the
staleness edge invisible rather than the race safe, and the 10-04 repair
(`b4df9bb`) deliberately declined that shortcut. **Do NOT** serialise by having
the Review hold a lock across its whole sitting — that would make a 20-minute
desk block every re-buy. Skip-and-log is the monotone move.
`T0.31` also owes a **re-run from a clean tree**; that is a separate unit and
the certificate is not healed by the dirt going away.

**B1. `scripts/lib_seal.sh` — the rescue sweep must not be gated on the sealed
file.** Ranked second only because B0 is already bleeding; this is the deeper
cause and it is what put the tree in the state B0 fired into.
`seal_output()` returns at `:317` and `:323` before the dirty-set sweep at
`:325`, so a dying run whose `$file` is clean gets **no sweep at all**. Hoist
the partition-and-sweep block so it runs on **every** `rc != 0` path, including
both clean-file branches. Evidence: 2026-10-09 06:57:09 logged *"already
carries a stale banner — leaving it"* and left that sitting's entire output
(`REVIEW_QUEUE.md`, `DECISIONS_NEEDED.md`) dirty for 24 h.
**Do NOT** change which files are swept or the `mtime >= run_start` predicate
that distinguishes this run's acts from another author's — that partition is the
74th audit's B1 and it is correct. Only the *reachability* of the block changes.
**Report the before/after on a forced `rc=124` with a clean `PROGRESS.md`.**

**B2. `scripts/lib_seal.sh:227` — `_seal_file_age_hours` must not count the
seal's own stamps.** It reads `git log -1 --format=%ct -- "$file"`, and
`stale_output()` commits that file, so each stamp zeroes the clock. Walk back to
the last commit touching the file that is **not** a seal stamp (the seal's own
commits are identifiable by author/message — `stale_output` writes them, not the
organ's agent, and says so at `:272`). Measured damage: 10-08 07:08:52 printed
*"only 24h old (cadence allows 25h) — still current, not stamping"* about a page
whose content was **96 h old and already bannered STALE**.
**Do NOT** raise `max_clean_age` (25 h) or remove the already-bannered guard —
both would hide the defect rather than fix it. This is a **reporting** repair:
it can only make the seal stamp MORE often, never less.

**B3. `experiments/run.py` — the `+dirty` re-buy message must not read CLEAN.**
All three rows in §1b record `HASH-SALT DIFFERENTIAL CLEAN (salt 1)` while their
commit field is `f24ac94+dirty`. The word is true of the salt differential and
false of the tree, and the row is what later readers believe. Append the dirty
marker to the recorded message when the stamp carries `+dirty` — e.g. a
`+dirty (N uncommitted file(s))` clause. **Strictly additive**: it changes no
verdict, moves no bar, and `run status`'s existing `! DIRTY STAMPS` reader —
which is correct and did its job — keeps its text.

**B4. `scripts/review_prompt.md:154` — the page should not be written last
(PROGRESS.md FTB item 5, 138th FTB 6, seconded by the Review 10-04, still
undone).** With B1 landed this stops being load-bearing, so **do B1 first and
then this**. The original reason for the ordering — not holding work dirty — no
longer applies: `docs/PROGRESS.md` is a `PROSE_DOCS` member and exempt from the
per-spec staleness bill. Yours under the 69th audit's B2 precedent.
**Do NOT remove the `INCOMPLETE`-row fallback.**
**And route it**: this item exists *only* on `docs/PROGRESS.md`, has **no
`REVIEW_QUEUE.md` row, no `DUE:`, and no instrument watching it** — which is why
it has survived six days on a page nobody can rewrite. It is the repair for the
mechanism that is costing certificates every two hours and it currently has no
clock.

**B5. Three of your eight steering orders are already discharged, and the page
cannot tell you.** `steering.read()` returns 8 items, all from
`docs/PROGRESS.md` — a page last rewritten 2026-10-04. Verified this audit:

- **item 1** (`T1.08` Steps 0+1) — `t108-pipeline-repair-has-no-design` is
  **ACTED 2026-10-08**; Step 2b executed 10-09 at `cff86c3`.
- **item 3** (additive `XL.01` pooled conjunct) — **done at `e9086ac`
  2026-10-05**; `xl_01_death_does_not_erase.py:106` records *"v3, 2026-10-05"*.
- **item 7** (`fieldwatch.py:102` `\bfinding\b` → `\bfindings?\b`) — **done**;
  the live source already reads `\bfindings?\b`.

Items **0, 2, 4, 5, 6** remain live; item 5 is B4. The Review hit exactly this
defect on 10-09 and fixed `PRIORITY.md` for it (`68874c7`: *"the builder's #1
priority pointed at work I closed one commit earlier"*) — but it cannot fix
`PROGRESS.md`, because only a completing run rewrites that page and none has
completed since 10-04. **Do not re-execute items 1, 3 or 7.** This is a reading
for you, not an order to edit the page — the page is the Review's.

## FOR THE OWNER

**1. The 28 GPU-hours died today, the arithmetic was published a day early, and
no organ had standing to prevent it. `D40` is on your desk and its `decide_by`
is today, 2026-10-10.**

`2026-W40` drew **1.971 h of 30 free Kaggle hours; 28.029 h expired Saturday
2026-10-10.** Fourth consecutive week — W37 1.379, W38 0.918, W39 1.072 —
**≈114.6 h unbought in four weeks.** The 147th audit computed the closed
arithmetic yesterday: the builder is the only organ that can spend them, and
`pace_gate` cannot legally release it until the usage week is 64 % elapsed
≈ 2026-10-12 20:30 UTC, about **2.4 days after the hours expire**. Nothing was
broken. Every organ obeyed its rules. The rules do not compose.

**I am not asking you to move the pace line, and no desk may.** The structural
fact is that the pace line is denominated in a usage pool of which **73 % is
another tenant's**, while the GPU allowance is denominated in a calendar week
that resets Sunday — so the two clocks can be, and now are, arranged such that
the project's free compute is unreachable by construction. This is the subject
`D40` already covers, it is armed, its default (v) is the status quo, and its
date is today. **If (v) fires, this recurs next Saturday with the same
arithmetic**, and I will report it again as realised rather than forecast.
Evidence: `experiments/gpu_budget.json` `weeks`; `/data/jack-logs/ladder.log`
2026-10-10T06:07:11.

**2. NO-DECISION, reported because you should not have to ask: three of your own
constitutional commitments have no live falsifier, and that is now eight days
old.** **smell**, **shelter/building** and **thermal (kills)** are CLAIM-DEAD —
`SM.02`/`SH.01` PARKED, `SM.03`/`SH.02` FORECLOSED, every park individually
legal and evidence-backed. The defect is compositional: nothing anywhere revives
the commitment behind the parks. This is `D42` (armed, `decide_by 2026-10-18`,
default (iv) HOLD — the three stay claim-dead and said so) and
`five-commitments-are-claim-dead-behind-foreclosures` (OPEN, `DUE: 2026-10-14`).
**Nothing is asked of you here that `D42` does not already ask.** It is repeated
because `D42`'s own price paragraph predicted that the daily report of it would
become wallpaper, and from where I sit it has: *"too cold kills him"* and *"he
builds a shelter"* are your words, and today the ladder cannot falsify either.

**3. One Tier-0 certificate's implementation was destroyed during this audit,
and no decision of yours is needed — you are being told because it is a loss,
not a risk.** At 06:45:06 the 2-hourly regate cron re-bought **`T0.31`** ("The
backlog reader cannot be quieted by tidying the backlog") while the Review was
mid-edit on `docs/REVIEW_QUEUE.md`, a file that spec genuinely reads. The record
says *"the code that ran was never committed, reconstructs from no commit and
was not preserved … it cannot be recovered by anyone."* **Nobody did anything
wrong** — both organs behaved correctly and the stamp that recorded the loss is
the system being honest. There is simply no interlock between a clerical cron
and a live desk, and at the 06:37 slot they overlap by construction. It is the
second such loss in three days (`T0.28`, 2026-10-08) and the repair is routed to
the builder as **B0**, with `T0.31` additionally owing a clean-tree re-run. The
verdict on this page is `INTEGRITY RISK` for this reason and this reason alone.

**4. A liveness report you are entitled to, with nothing to rule on.** The
**Review has not completed a run since 2026-10-04** — `rc=124` on 10-05, 10-06,
10-07, 10-08 and 10-09, and **4 of its last 16 runs** exited 0. It is doing real
work in those twenty minutes (the whole of §6 and §7 is its output) and losing
only its page — but §3 is the cost of that, and it is no longer only a page.
The builder has been dark **41 slots / 41.6 h**, legally, paced out by another
tenant's usage. `demonstrated` **107/257**, unchanged since 2026-10-08. All four
organs fired within cadence. Sections 1 and 2 are clean: **nothing has been
loosened and no PASS is unsupported.**
