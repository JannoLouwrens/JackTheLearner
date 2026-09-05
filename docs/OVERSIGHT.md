# OVERSIGHT — 74th audit, 2026-09-05 12:37–13:0x UTC (at `6e338ee`)

## VERDICT: DRIFTING — the Review died at max turns this morning, and the machine that exists to stop a dead run's verdict from being believed sealed its **report** while its **actions** were committed unmarked six hours ago: a new owner decision, a shrink-only ratchet lowered 3→2, the week's only queue disposal, and the 122-line priority block that has steered every builder slot since. Separately, the ownership map I ordered yesterday returned its first reading with **3 of its 5 `mention-only` rows misclassified**, two of them the flagship rows the whole exercise was for.

Say the clean part first. It is large, it was checked mechanically, and none of
it was inherited from this morning's audit.

**Section 1 — ledger integrity: clean, third consecutive audit.** All **104**
PASS rows resolved by hand: **0** with no implementation reachable through
`protocol.module_path_for`; **0** whose `commit` field fails `git cat-file -e`;
**0** with no `control` declared in the registry; **0** whose declared control
has no token present in the test source. Two rows carry empty `control_metrics`
— `T0.01` and `T0.10` — and both declare `"NONE, BY DECISION (52nd audit B5)"`,
so the absence is pre-registered rather than silent. **No finding.**

**Section 2 — thresholds and controls: clean, seventh consecutive week.** I ran
my own paired constant-diff over 7 days of `registry.py`,
`registry_expansion.py` and `experiments/tests/` (collect every
`NAME = <number>` on a `-` line and on a `+` line; report names whose value set
changed). **Ten constants moved. Nine moved in the strengthening direction:**
`N_PROPERTIES` up repeatedly to 15, `N_LIVES` 16→32, `N_EVAL` 48→120 and →800,
`N_DECISIONS` 3200→4800/6000/20000, `LIVES_PER_ARM` 4→16→48, `COORD_MIN`
0.55→0.70, `COORD_MARGIN` 0.20→0.35, `STEPS` 300→500, `TEMP` 0.25→1.0. **One
moved downward**: `DECAY_MIN` 1.5→1.25 (`44f24c4`, `T2.09`) — the `PILOT`-marked
placeholder being frozen for the first time from disjoint seeds 7/90, read in
full by the 72nd and 73rd audits and justified in its own commit. **Zero
constants moved in the six hours since the last audit.** No control was deleted,
no `_check` gained an `or`, no seed count was reduced, no assertion was removed.
**No finding.**

**Sections 6 & 7 — decisions and bakeoff hygiene: no finding.**
`decisions --check` rc=0: 10 armed, **0 UNDECLARED, 0 MEANS-ESCALATED, 0
OVERDUE, 0 VANISHED-OWNER-ASK, 0 UNROUTED-OWNER-ASK**. Nothing is on the owner's
desk that a measurement could settle, so there is nothing to seize under rule 3
and — for the first time in this organ's recent record — **nothing to arm,
because nothing is undeclared.** Both live owner-asks on `PROGRESS.md` are
attributed (`#1`→`D23`, `#3`→`D21`). `champions --check` rc=0 with **0
ARENA-MISSING**; the two `UNVERIFIED VERDICTS`, three `TRIGGER DEBT` seats and
two `UNFALSIFIABLE` seats are at their declared baselines. `coverage` exits **2**
on the standing routed red (`claim_dead` 4, `goal_unrunnable` 7) — both carry
dated queue rows.

**Section 5 — compute honesty: no finding.** CPU today **9,302.96 s of 57,600**,
of which **9,205.09 s (99.0%) is `SO.07`'s registered claim run**; the three
instrument re-buys behind this morning's B1–B3 cost **22.11 s in total**
(`T0.21` 27.39−9.15, `T0.28` 66.61, `T0.31` 3.87). That is the second
consecutive day whose composition refutes the day-meter's first-day reading.
GPU: **W35 at 19.20 h of 30, 10.80 h expiring at midnight** with every GPU cost
class empty, pilot-blocked or holding only settled VOIDs. Letting it expire is
correct and I endorse it unchanged — this is inventory, not uptime.

**Section 4 — the builder is alive and doing exactly what it was told.** Six
iterations since 06:37, **six ended rc=0**, zero `PACING:` skips, `week:all
models` 11% at the last read. Five of the six were verified-empty-board slots
ended early under the standing rule, and one of them (11:07) caught its own
false `rc=0` receipt from a pipe and turned it into a strengthen-only lesson.
That is good conduct. The finding in section 3 below is about what the loop was
steered *by*, not about how it behaved.

Now the findings, ranked by damage to the trustworthiness of the map.
**None of them is a false PASS.**

---

## 1. THE SEAL QUARANTINES A DEAD RUN'S REPORT AND NOT ITS ACTS. THE 09-05 REVIEW DIED AT MAX TURNS WITH FIVE DIRTY FILES; ONE GOT A BANNER AND FOUR WERE COMMITTED AS ORDINARY WORK

At **06:52:20** the daily Review hit `Error: Reached max turns (60)` and exited
rc=1 (`/data/jack-logs/review.log:480`). `scripts/lib_seal.sh` did its job
exactly as written: it stamped `docs/PROGRESS.md` with

> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.** … any verdict, any
> section claiming "no findings", and any instrument table in it are UNVERIFIED.

and committed it as `7275224`. **That commit touches one file.**

The dying run had left **five** files dirty. At **07:09** the builder committed
the other four as `e034b94`, *"Inherit the 09-05 DAILY Review's uncommitted
edits (its run died rc=1 mid-report)"* — carefully, with `decisions --check` and
`champions --check` verified rc=0 before committing, and with the provenance
named in the message. **The builder's conduct is not the finding.** The finding
is that nothing in this repository marks those four files as the output of a run
whose own page says its work is unverified:

| file | what outlived the dead run, unbannered |
|---|---|
| `docs/DECISIONS_NEEDED.md` | **`D23`, 136 new lines** — a live owner decision, `class: goal`, `decide_by 2026-09-11`, whose default now **fires by silence**. Its entire evidentiary basis is `PROGRESS.md`'s `FOR THE OWNER #1`, which carries the UNVERIFIED banner. |
| `docs/CHAMPIONS.md` | the *Language grounding* seat `ARENA: NONE → LG.04, LG.05, LG.06`, which moved the **shrink-only** `UNFALSIFIABLE` ratchet **3 → 2**. A ratchet that may never grow was lowered by an incomplete run. |
| `docs/REVIEW_QUEUE.md` | `champions-language-grounding-arena` closed **ACTED** — **the only disposal in seven days**, and therefore *half* of the `disposed 2 (0.29/cycle)` figure every reader now quotes — plus `t027-preserved-failimpl-as-artifact` re-dated **09-05 → 09-07**, moving a row off the day it came due. |
| `scripts/ladder_prompt.md` | **122 lines** replacing the builder's priority block `1''/2''` with `1'''/2'''`. |

That last row is the live one. **Five consecutive builder slots (08:07, 09:07,
10:07, 11:07, 12:07) have executed it and re-confirmed its handoff verbatim** —
the "verify emptiness and stop early" rule, the SO.08 ordering, the D15/D16/D21
sequencing. Roughly five hours of loop time steered by an instruction whose
author never finished its own checklist.

**Why this is rank 1.** `lib_seal.sh`'s own scar-story is precise about the harm:
the 49th audit's dying run left `OVERSIGHT.md` reading `VERDICT: ON TRACK` and
*"the log and the file disagreed, and nothing joined them."* The repair joined
them — **for one file**, because `seal_output <rc> <file> <organ>` takes one
path. But a Review's job is *not mainly to write its page*; it is to dispose
queue rows, arm decisions, re-mark champion seats and steer the builder. Those
acts are its product, and they are precisely the part the seal cannot reach.
The banner is now doing something worse than nothing on its own terms: it tells
a reader that this run's conclusions are unverified while four files carrying
this run's conclusions sit in the tree with no banner at all, and one of them is
on the owner's desk with a clock running.

I am not asking for `D23` to be withdrawn or the seat edit reverted — I checked
both and both are defensible on facts I can verify independently (`fail_unowned`
did go 4→0; `drain` is UNBOUNDED; `LG.04`/`LG.05`/`LG.06` do exist). **The
finding is that no reader in this repository could have told me that, and I had
to check by hand.** See **B1**.

---

## 2. `fail_unowned`'s OWNERSHIP MAP IS FIVE HOURS OLD AND MISREADS 3 OF ITS 5 `mention-only` ROWS — INCLUDING TWO OF THE FOUR THE WHOLE EXERCISE WAS FOR

My own B2 yesterday asked for this, in these words: *"make the strength of
ownership readable… the number stays at floor; the map stops implying repair."*
The builder shipped it at 07:14 (`34f3f48`), correctly, with a fixture battery
(`Z.7`, `Z.8`) and no change to the count. `run status` now prints:

```
fail_unowned = 0
  owned: 1 repaired_by, 1 disposed, 15 queue-row, 5 mention-only
```

**The five `mention-only` rows are `NE.01`, `T2.05`, `T4.02`, `W.1`, `W.2`.**
Two of those — `T2.05` and `T4.02` — are members of the batch of four that the
72nd audit routed at 01:19 yesterday to take `FAIL-UNOWNED` from 4 to 0. Their
three siblings read `queue-row`. **All four rows were written in one commit, one
minute apart, by one author, with the identical `DUE: 2026-09-13`.**

**The mechanism, at `experiments/coverage.py:1179-1194`.** `_owned_by_dued_row`
walks the queue and terminates a row's block at the first non-indented, non-empty
line:

```python
if raw.strip() and not raw[:1].isspace():
    if _hit(block): return True
    block = []
```

A queue row's shape is `ROUTED:` / indented `DUE:` / blank line / **flush-left
evidence paragraph**. So the block that gets searched is *the two header lines
and nothing else*, and every row's body is skipped by `if not block: continue`.
The spec id therefore counts as owned **only if the author happened to type it
into the `DUE:` sentence**:

| row (all `DUE: 2026-09-13`, all from `52dcf9e`) | id in the `DUE:` line? | form |
|---|---|---|
| `xl01-death-and-retry-has-no-reachable-repair-path` | yes — *"NOT 're-run XL.01'"* (:2235) | `queue-row` |
| `t215-heldout-language-routing-…` | yes — *"a disposition for T2.15's FAIL"* (:2299) | `queue-row` |
| `t205-world-model-loses-to-the-ridge-reference` | **no** — *"a disposition for the fast/slow world-model fixture"* (:2261); `T2.05` appears at :2263, in the body | **`mention-only`** |
| `t402-touch-drowns-audio-at-the-fusion-boundary` | **no** — *"the fusion-balancing redesign…"* (:2280); `T4.02` appears at :2282, in the body | **`mention-only`** |

The row slug (`t205-…`) is lowercase-and-dashes and cannot match the id regex,
so the row's own title never helps.

**And a second facet, which is a genuine disagreement between two readers.**
`NE.01` has a real row — `ne01-occlusion-knife-edge` (:229) — `HELD` and paying
with `BLOCKED-BY: w0-too-shallow`. `REVIEW_QUEUE.md`'s own written rule (:43)
says a hold *"must pay for it with a `DUE:` **or** a `BLOCKED-BY:`"*, and
`review_queue.py` enforces exactly that (`HOLD-WITHOUT-A-CLOCK`). But
`_owned_by_dued_row` accepts **only** `DUE:`. Its docstring claims the opposite
guarantee — *"the declaration test is its `_DECL` regex, not a fresh one, so the
two readers cannot drift apart on what a declared clock looks like"* — and they
have drifted, on the one form the other reader calls legal.

Only `W.1` and `W.2` are correctly weak: their rows are `OPEN` with no clock of
any kind.

**Why this matters and why it is rank 2 rather than rank 1.** The count is
unaffected — a `mention-only` still counts as owned, so `fail_unowned` stays at
its floor of 0 and no ratchet is wrong. The damage is to the map, which is the
whole thing B2 asked for. A reader today cannot distinguish *"weak because the
row has no clock"* (`W.1`, `W.2` — true, and the reading that should drive
Sunday's triage) from *"weak because the author wrote a description instead of an
id"* (`T2.05`, `T4.02`) or from *"weak because two of our own tools disagree
about what a valid row is"* (`NE.01`). The number `5 mention-only` is already
committed to `ratchet_readings.json` as `review_queue`-adjacent evidence, and the
first thing anyone will do with it is decide which negatives are least owned.
**Its top two entries are wrong, in the direction that makes a genuine dated
promise look like an unowned failure.**

This is the third audit running to find the same shape and the second to find it
on an instrument this organ itself ordered: `piled_on` counted 0 of 4 yesterday;
`fail_unowned`'s form-map misreads 3 of 5 today. Both fixture batteries pass,
because in both cases the fixture put the datum where the code looks. See **B2**.

---

## 3. TOMORROW IS THE WHOLE PROJECT'S SINGLE POINT OF FAILURE, AND TODAY THE SAME ORGAN DIED AT MAX TURNS ON A THIRD OF THE WORK

`scripts/review.sh:56-58`: `TURNS_PER_MIN=3`; Sunday → `FULL`, 40m / **120
turns**; otherwise `DAILY`, 20m / **60 turns**.

**The base rate, from `/data/jack-logs/review.log`:**

| | |
|---|---|
| DAILY runs since 08-13 | 13 |
| completed rc=0 | 11, in **8–13 minutes** each |
| died `Reached max turns (60)` | **1 — today, at 15 minutes** |
| FULL runs ever attempted on cron | 4 (all Sundays) — **4 deaths, all `Reached max turns`** |
| FULL runs ever completed | **1**, the 08-31 **rehearsal**: launched by hand at 19:11, finished in 10 min, *"It fits… with turns to spare"* |

The budget repair on 08-31 was real and it was right. But the one FULL that has
ever fit was measured against a queue of roughly 20 live rows with **no dated
docket**. Tomorrow's carries:

- **six live rows dated 2026-09-06** — both `d10-*` gate rows (which release a
  ~16.18 h dispatch into W36's 30 free hours), `w0-too-shallow` at **12 days and
  eleven instruments**, `lt01-c2-body-cannot-rise`, `lc07-checkpoint-branch`,
  `cross-organ-doc-race-voids-certificates`;
- **Part 2** at its stated minimum of 8 specs, plus the **anatomy audit** and the
  **completeness audit** — the one that once found smell, taste, voice and
  body-schema with zero specs among 136;
- **`D21`'s default**, which fires at 00:0x and *commands* that same run to take
  the **W1 spec-family design as the first design item**;
- against a live board of **38 rows**, up from ~20 at the rehearsal.

That is not 2× a DAILY. Today a DAILY exhausted 60 turns; tomorrow's FULL gets
120 for a docket several times larger.

**And findings 1 and 3 compose, which is why this is on the page today rather
than as a note.** The likeliest death for this organ is a *late* one — the seal's
own docstring says so. A late death tomorrow banner-seals `PROGRESS.md` and
leaves its dispositions of `w0-too-shallow`, the two `d10-*` rows and the W1
design sitting dirty and unmarked, for the 00:07-onward builder to inherit
exactly as it inherited `D23` this morning. **The single most consequential
Review run this project has scheduled would fail in the one mode its safety net
does not cover.** See **B1** and **B3**.

---

## 4. SECTION 3 — DRIFT: NOTHING SERVES NO GOAL SENTENCE, AND NOTHING SERVED JACK

Every non-journal commit since 06:37 traces to `GOAL.md` through this organ's own
B-items or the Review's: `e034b94` (inheritance), `34f3f48` (B1/B2/B3),
`78ddc01` + `75fb1b3` (`T0.21`/`T0.31` re-buys the instrument edits staled).
**No drift.**

**And no science.** `demonstrated` has read **104 → 104 across six consecutive
iterations** since 07:16. Five of those were correct empty-board slots that
billed nothing, which `PROGRESS.md`'s own FOR THE BUILDER item 2 defines as
correct conduct and which I endorse — manufacturing work would have been worse.
But the honest statement of where the ladder is:

- last **credited** claim about Jack: **`LG.02`, 2026-09-02 — three days ago**;
- `T2.01` FAIL, **frees 35 / blocks 38**, implementation unchanged **26 days**;
- `one brain / unison`: 25 specs, **1 pass**. `fast/slow`: 8 specs, **0 pass**,
  five welded behind `LC.03`. `sleep`: 5 specs, **0 pass**. `curiosity`: 12
  specs, 2 pass. Four claim-dead commitments (smell, balance, shelter, thermal);
- the **one** fresh dispatch on the entire board is `SO.08`, and it is held by a
  clock until midnight.

---

## FOR THE BUILDER

**B1 (rank 1). `seal_output` must cover the run's whole dirty set, not one
file.** `scripts/lib_seal.sh`'s signature is
`seal_output <rc> <output-file> <organ>` and its DRAFT branch stamps and commits
exactly that path. A dying organ leaves *every* file it was mid-edit dirty, and
this morning four of the five went out unmarked. The repair keeps the existing
banner behaviour for the organ's own report and adds a second, cheaper act for
the rest:

1. After sealing `<file>`, enumerate the run's other dirty paths —
   `git status --porcelain` minus `<file>`, minus anything already staged by
   another organ if you can tell — and **commit them in one path-scoped commit
   whose message names the rc, the organ and the sealed report**, so
   `git log` joins them the way the banner joins the report to the log. Do not
   discard them; the 49th-audit lesson (*"an uncommitted report in a shared tree
   is one `git clean` from gone"*) applies with equal force to a disposition.
2. **Append one line to the sealed report listing those paths**, e.g.
   `> Files this run also left dirty, committed unbannered: docs/CHAMPIONS.md,
   docs/DECISIONS_NEEDED.md, docs/REVIEW_QUEUE.md, scripts/ladder_prompt.md.`
   That single line is what would have told me this morning, without a hand
   check, that `D23` and a 3→2 ratchet move came from an incomplete run.
3. **Falsifier, and it is the point:** extend `T0.34` (or `T0.31`, whichever
   owns the seal) with a property that constructs a dirty tree of **N>1** files,
   calls `seal_output` with rc=1 on one of them, and **asserts on the OTHER
   files** — that they are committed and that the report names them. Verify it
   FAILS against today's code before you fix it, exactly as you did for `P14b`.
   The class, not the tidy example.

Explicitly **not** asked for: any change to what an organ may edit, any
quarantine that would block a builder from inheriting real work, or a gate. The
edits this morning were good work and the builder was right to keep them. What is
missing is a mark, not a wall.

**B2 (rank 2). Fix `_owned_by_dued_row`, both facets, and ratchet the corrected
breakdown.** `experiments/coverage.py:1179-1194`.

- **Facet (a) — the block is two lines long.** A row's evidence paragraph is
  flush-left, so `block = []` fires before the body is ever searched and the id
  counts only if it was typed into `ROUTED:`/`DUE:`. Extend a row's block to the
  next column-0 `ROUTED:` line (the file's real record boundary), **or** — better,
  because it is exact and cannot over-attribute — additionally match the row's own
  **slug** against the spec id (`t205-…` ↔ `T2.05`, `t402-…` ↔ `T4.02`), which is
  how a row actually declares its subject. If you extend the block instead, check
  the over-attribution direction: the `xl01` row's body names `NE.01` and `NE.08`,
  and neither should inherit `xl01`'s clock.
- **Facet (b) — `BLOCKED-BY:` is legal payment and this reader rejects it.**
  `REVIEW_QUEUE.md:43` and `review_queue.py`'s `HOLD-WITHOUT-A-CLOCK` both accept
  `DUE:` **or** `BLOCKED-BY:`; `_hit` accepts only `DUE:`. That is why `NE.01`
  — a properly `HELD` row with a named blocker — reads `mention-only`. Either
  accept both and print a distinct form (`held-on-blocker`), or state in the
  docstring why a blocked hold is deliberately weaker. **Do not** silently make
  the docstring's "cannot drift apart" claim true by deleting it.
- **Expected corrected reading:** `mention-only` **5 → 2** (`W.1`, `W.2` — the
  two genuinely clockless rows), `queue-row` 15 → 17, `NE.01` to whatever form
  (b) settles on. Record the corrected breakdown alongside `fail_unowned` in
  `ratchet_readings.json`; the count stays 0 and **must not move**.
- **Falsifier on the class:** the fixture battery passes today because `Z.7`
  puts the id where the code looks. Add a fixture row whose id appears **only in
  a flush-left body paragraph** under a `DUE:`-carrying header, and one whose
  only clock is a `BLOCKED-BY:`. Verify both FAIL first.

**B3. Price tomorrow's FULL against its docket before it runs, and make a late
death survivable.** Two things, and the second matters more than the first.

1. **Survivability first, and it is free:** instruct the FULL (in
   `scripts/review_prompt.md`) to **commit each disposition as it is made** —
   one path-scoped commit per queue row disposed, per seat re-marked, per
   decision armed — rather than leaving them dirty until the page is written.
   Then a max-turns death costs the *page*, not the *work*, and B1's banner has
   less to say. This is the cheapest possible mitigation and it needs no budget
   change.
2. **Then the budget.** `TURNS_PER_MIN=3` gives FULL 120 turns; today a DAILY
   spent 60 on a fraction of tomorrow's docket, and four of the four FULL runs
   that ever fired on cron died at max turns. The 08-31 precedent is that this
   is builder work at the Review's request. Raising an **organ's turn budget** is
   not loosening a science threshold and nothing in `SYSTEM.md` treats it as one
   — but say so explicitly in the commit, with today's death and the 4/4 Sunday
   record as the measurement, so no future audit reads it as a silent loosening.
   If you would rather not touch the number the day before it fires, item 1 alone
   is worth shipping tonight.

**B4. Two corrections to the 00:07 handoff you have re-confirmed five times
today — both inherited from the sealed page, and both wrong in the direction
that could cost you `SO.08`.**

- **`SO.08`'s constraint is 3,600 seconds of BILLING SLACK, not a 60-minute wall
  clock.** `PROGRESS.md` wrote *"the window is one hour wide"*; your 08:07
  journal turned that into *"a 60-minute launch window at 09-06 00:07"* and four
  slots have repeated it. The actual arithmetic is
  `t0_33_cpu_budget.py:173`: `WORST_LEGAL_CHILD_S = 54000.0` against a 57,600 s
  day, so what forecloses `SO.08` is the day's **other billing exceeding
  ~3,600 s** — at any hour. A unit that spends ninety minutes on paperwork has
  **not** missed the window; a unit that runs one 2-hour CPU spec first **has**.
  Do not skip `SO.08` because a clock you were handed says the hour is over.
- **`D15`'s default is not "seconds of paperwork".** `PROGRESS.md` FOR THE
  BUILDER 1 describes all three defaults that way; `D16`'s and `D21`'s are, and
  `D15`'s is not. Read it: it adds a `pace_gate` to `overseer.sh` with a
  first-audit-of-day exemption, **and** makes *every organ script* append
  `{organ, ts, pct, model_pct, phase}` to `usage_ledger.jsonl` at start and end.
  That is four scripts and a new log format — and `scripts/ladder_loop.sh` is in
  **`T0.33`'s `IMPL_DEPS`**, so firing it stales the CPU-budget certificate that
  the day meter admitting `SO.08` is built on. Order it: fire `D21` (the
  same-day race, must beat 06:37) and `D16` (a no-op that keeps `T0.27` red),
  **start `SO.08`**, then fire `D15` and re-buy `T0.33` (~12 s) while it runs.
  If that ordering is wrong, say why in the journal — but do not fire a
  four-script change ahead of the only startable claim about Jack on the board
  because a dead run called it paperwork.

**B5. Standing prohibitions, carried forward unchanged and all still live:** do
not re-dispatch `D1.0` (gate design owed at tomorrow's FULL; an unchanged re-run
is a 16.17 GPU-hour seed-lottery redraw); `HR.1`–`HR.4` stay `D19`-held to 09-14,
no corpus fetch; `HR.6` stays blocked behind `HR.5`; `LF.01` attempt 2 waits for
the 09-09 design and `FIXTURE_VOID_CAP=3` is not permission; do not re-stagger
the 09-06 docket by hand a third time; do not re-date any row to shrink
`piled_on`; let `W35`'s 10.80 Kaggle hours expire tonight. The CPU-accountant
prohibition stands in the **narrowed** form the Review corrected it to: repairs
that make the meter refuse fewer runs or print more honestly need no permission;
new accounting *surface* does.

---

## FOR THE OWNER

**1. `D22` is still the decision everything routes through, its default fires
2026-09-08 as "the rule stands", and the measurement moved again in one
direction only.** Third consecutive day of the same table, taken from the live
tools this hour:

| | 09-04 (Review) | 09-05 06:37 (73rd) | 09-05 12:37 (this audit) |
|---|---|---|---|
| live queue rows | 33 | 39 | **38** |
| arrived / disposed, trailing 7 d | 30 / 1 | 36 / 1 | **36 / 2** |
| drain | UNBOUNDED | UNBOUNDED | **UNBOUNDED** |
| dates carrying more than the measured capacity of 1/cycle | 6 | 7 | **7** |
| rows dated onto an already-full day (`piled_on`, corrected) | — | 22 | **22** |

The single disposal that moved that column from 1 to 2 was made by the Review run
that died — see finding 1. **Silence to 09-08 is not neutral**: at the measured
arrival rate it costs roughly 17 further net rows, and every startable spec on
the board sits behind that desk. The default is the only legal one available, so
the fork remains exactly as the Review put it to you, now priced three times.

**2. One minute of crontab, carried from yesterday and now demonstrated rather
than argued.** `37 6 * * * review.sh` and `37 */6 * * * overseer.sh` fire in the
same minute. Yesterday the 06:37 audit could only read the previous day's page;
**today it read a page that was, at that moment, being written by a run that was
about to die**, and this 12:37 audit is the first organ in the chain to see the
`INCOMPLETE RUN` banner at all — six hours and five builder slots after the
edits it covers were committed. Changing `overseer.sh` to `47 */6 * * *` (or
`review.sh` to `27 6`) fixes it. `crontab` is outside the repo and outside what
any agent here may touch, so it needs your hand. **No urgency, no risk, and no
agent can do it.**

**3. NO-DECISION — a note on what tomorrow costs if it goes wrong.** Nothing to
rule on; recorded because it is the largest scheduled risk on the board.
Tomorrow's FULL Review is the sole owner of six dated rows, of the `W1` design
that `D21`'s default will command it to take first, and of the two `d10-*` gate
rows that release a ~16 h dispatch into W36's 30 free GPU-hours. Four of the four
Sunday FULL runs that have ever fired on cron died at max turns; the only one
that ever completed was a hand-launched rehearsal against half today's queue. I
have asked the builder for the cheap half of the mitigation (commit each
disposition as it is made, so a late death costs the page and not the work) and
left the budget question to it with the measurement attached. If it goes wrong
you will see it in `docs/PROGRESS.md` as an `INCOMPLETE RUN` banner tomorrow
morning — and, until **B1** ships, you will **not** see it on whatever else that
run touched.

**4. The honest paragraph — are we closer to a curious humanoid that climbs the
ladder than we were yesterday?** **No. We are exactly where we were, and today
that was the right answer.** `demonstrated` has read 104 for six consecutive
iterations. Five of those were slots that read the board, verified it empty with
their own receipts, wrote down what they checked, and stopped — which is the rule
the Review wrote and the correct behaviour when there is nothing to run. Nothing
was manufactured, nothing was billed, and one slot caught its own false green
receipt and turned it into a lesson. That is a healthy loop with an empty board,
and the board is empty because every path forward runs through a design desk that
disposes one row a week.

What I want on the record is the shape, because it is now the **fourth**
consecutive audit to find it and the **second** in which the instrument at fault
is one this organ ordered the day before. `piled_on` counted 0 of 4. `fail_unowned`
went to floor in six hours, and its repair — the ownership map — misreads 3 of
its 5 weakest rows on its first reading. The seal, built precisely so a dead
run's confident verdict could not be believed, watched a dead run's four *acts*
walk out unmarked six hours ago, including one that lowered a shrink-only ratchet
and one now sitting on your desk with a clock. Every one of these was built
honestly, by an organ doing its job, and every one of them reads green through
the exact behaviour it exists to catch. The counterweight is not more
instruments — this project already has more instruments than claims about Jack.
It is that each new one gets a falsifier that fires on the **class** and not on
the tidy example, which is `T0.31`'s P4/P5/P6 rule, already written down, and the
reason every B-item above asks for a falsifier verified to fail *before* the fix.
