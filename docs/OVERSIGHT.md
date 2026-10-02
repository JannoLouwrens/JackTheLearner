# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-10-02 06:4x–07:1x UTC — the 136th audit.** Six hours after the 135th, on
cadence. My window is the builder's slots `01:07` through `06:07`: **six slots,
ZERO `rc=0`, six PACE-SKIPs.** Demonstrated moved **107 → 107**.

Instrument exit codes, re-derived this sitting from source: `coverage` **2**,
`decisions --check` **1**, `champions --check` **0**, `run status` **2**,
`run review-queue` **2**, `review_liveness` **1 (FAILED)**.

> **THE REVIEW IS MID-SITTING WHILE I WRITE, and it is working on the rows this
> report is about.** `scripts/review.sh` (pid 3236454) started at 06:37 beside my
> own `overseer.sh` — the collision `LESSONS.md` records. `HEAD` moved from
> `a3c919f` to `4f53fb0` during this audit (six Review commits, 06:41–06:46), and
> `docs/REVIEW_QUEUE.md` is dirty in the tree with that desk's live edits. **I
> staged none of it and I `git add` by name.** Every queue number below is
> stamped with the `HEAD` it was read at, and §RANK 7 is written as a statement
> about the **2026-10-01** sitting, not about the one happening now — which is
> disposing exactly the right rows and is credited there.

Ratchet delta, quoted from `run status`'s own SLOT LINE before anything here was
recorded: **`dark_slots` 0 → 28; `review_queue_violations` 14 → 13;
`review_queue_piled_on` 4 → 6; `review_queue_net_arrivals` 32 → 31;
`review_queue_violation_forms` HOLD-ON-A-RESOLVED-BLOCKER 8 → 7. No counter
refused to compute. Floors: 4 ABOVE (`dark_slots`,
`decisions_default_action_expired`, `pass_on_dead_dependency`, `unreachable`), 0
BELOW, 0 UNVERIFIED.** Three of the four breaks are inherited with written causes
I re-derived rather than trusted. `dark_slots` is the 135th audit's finding grown
**seven-fold inside one day**: it read 4 then and 28 now.

Ledger: **PASS 107 / FAIL 33 / VOID 16 / BLOCKED 1** over 157 rows with a verdict
against **255 specs** → **42.0 %**, unchanged in 24 h. Rework: **127 of 157 rows
are attempt > 1 (80.9 %)**.

**What I did NOT do, named rather than omitted.** I re-ran no spec, dispatched
nothing, and every number here is read from the committed ledger, from git, from
`/data/jack-logs/`, or produced by a read-only instrument. **I armed no decision:**
`decisions_undeclared` reads **0** — there is nothing armable on the register, and
manufacturing an entry to satisfy the per-audit quota is the disease the quota
exists to prevent. I re-derived that rather than copying the 134th and 135th
audits' identical refusals. **I fired no default:** `D37` is the only armed entry
and its date, 2026-10-04, has not arrived. I appended an EVIDENCE ADDENDUM to the
already-open `D30` rather than opening a duplicate entry.

---

## VERDICT: DRIFTING — and the new thing is that **the pace line that was built to keep the loop awake when the free GPU quota expires was calibrated against the wrong week.** Its own header says *"by Friday the line is ~62 %, Sunday ~81 %, and the loop is still awake when the GPU quota expires."* It is Friday; the line is **42 %**. On Sunday — the day ~28.9 free Kaggle GPU-hours die — it will be **63 %** against a meter at 79 %. The builder has been dark **28 consecutive slots** and cannot legally run again before **Tuesday 2026-10-06 09:06 UTC**, while the account sits **11 points from the 90 % hard stop that pauses every organ**, roughly **six hours away** at the measured external rate.

Why not `INTEGRITY RISK`: the ledger's mechanical integrity held on every check I
ran (§1 — 107/107 implementations resolve, 107/107 recorded commits resolve, 0
PASS rows with an undeclared control, 0 PASS rows with empty `control_metrics`
except the two explicit `NoControlByDecision` refusals), and **no threshold moved
in the loosening direction in seven days** (§2, verified by my own extraction, not
inherited). Nothing below touches a capability claim.

Why not `ON TRACK`: **six of six slots in my window produced nothing**, the
previous audit's four FTB orders have had **zero live builder slots to be read
in**, and the ladder has not moved in five consecutive audit windows.

---

## RANK 1 — `PACE_FLOOR`/`PACE_CAP` were calibrated for a week that starts Monday 00:00. The meter's week starts **Wednesday 11:59 UTC**. The line is therefore ~20 points tighter than designed across the whole back half of every calendar week — including the Sunday the free GPU quota expires, which is the one day the gate was built for.

`scripts/lib_usage.sh` is a pure function of the clock and it is computing
correctly from a correct input. **The defect is in the two constants and in the
comment that justifies them.**

The gate's own design note (`:59-60`):

> `PACE_FLOOR=25` buys the week's opening burst; **by Friday the line is ~62 %,
> Sunday ~81 %**, and the loop is still awake when the GPU quota expires.

Those two numbers reproduce **exactly** under a Monday-00:00 week, and under no
other:

| | design (Monday 00:00 week) | actual (meter's Wed 11:59 week) |
|---|---|---|
| Friday | elapsed 57 % → line **63 %** | elapsed 25 % → line **42 %** |
| Sunday | elapsed 86 % → line **81 %** | elapsed 57 % → line **63 %** |

The meter's reset is not inferred — `scripts/claude_usage.py` prints it:
`week:all models … resets Oct 7, 11:59am (UTC)`, and Oct 7 2026 is a **Wednesday**.
`--week-elapsed` returns **25** right now; a Monday reset would return ~75. The
gap is 59 h of week, i.e. 35 points of `elapsed`, i.e. `0.65 × 35 ≈ 23` points of
line. **The real Sunday line equals the design's Friday line.**

**Why this is RANK 1 and not trivia.** The pace line's whole stated purpose, in
its own header, is the failure it has now presided over three weeks running:

> *"30.9 free GPU-hours have died in two weeks with no agent awake to dispatch
> them — on a project whose owner has ruled free compute only."*

At the **designed** Sunday line of 81 % the builder would run this Sunday
(`79 < 81`). At the **actual** line of 63 % it will not (`79 > 63`), and
2026-W39's ~28.9 unspent free Kaggle GPU-hours expire that weekend. The gate was
built to prevent exactly this and its calibration guarantees it.

**This is NOT mine to repair and must not be repaired casually.** Raising
`PACE_FLOOR` *widens* what the builder may spend, so it cannot be a default
(`D26`'s reasoning) and it cannot be a builder edit over its own constraint. It
belongs where it already sits: `D30` option **(i)**, the owner's. What I have
added is the arithmetic, in the addendum.

## RANK 2 — 28 consecutive dark slots, and the account is ~6 hours from a hard stop that pauses EVERY organ. Nothing in `experiments/` reads the usage meter.

Measured, not reasoned. `/data/jack-logs/ladder.log`: **unbroken `PACING:` skips
from `2026-10-01T03:07` through `2026-10-02T06:07`** — 28 slots, zero `rc=0`,
zero failures. `run status` reads `dark_slots = 28 !! MOVED +28 since 2026-09-30
(was 0)`, `!! ABOVE its declared floor 0`. The last `rc=0` was **28.4 h ago**
(`02:07` on 10-01, the `LG.14` registration).

Live, this sitting:

```
week:all models   79%        (scripts/claude_usage.py --pct)
week:Fable        52%        (not the gate)
week-elapsed      25%        line = 25 + ceil(65*25/100) = 42%
.usage-resumed    ABSENT
```

**Two dates, both computed from the source constants:**

- **First legal builder slot: `2026-10-06 09:06 UTC` (Tuesday).** Release needs
  `79 < 25 + ceil(0.65·e)`, i.e. `e ≥ 84 %` of the week; 84 % of the week that
  began 2026-09-30 11:59 is that timestamp. **This holds only if the meter rises
  no further**, so it is a floor on the blackout, not a forecast.
- **90 % hard stop: ~`13:00 UTC today`.** The external draw is measured, not
  estimated: `usage_ledger.jsonl` records `pct 37` at 2026-10-01T06:57 and
  `pct 79` at 2026-10-02T06:37 — **42 points in 23.7 h = 1.77 pts/h** — across a
  span in which this project ran two desk sittings and nothing else. 11 points
  remain.

**What the hard stop does, read from source rather than assumed.**
`usage_gate` is called by `scripts/ladder_loop.sh:111`, `scripts/overseer.sh:47`,
`scripts/review.sh:57` and `scripts/field_watch.sh:32`. With no `.usage-resumed`
it logs *"STOPPED at N% weekly usage — all agents paused until the owner
resumes"* and returns 1. **Only `scripts/regate.sh` survives, and it calls no
model.** So at ~13:00 UTC the builder, the Review, this organ and the field watch
all stop, and the only thing that restarts them is a file the owner writes.

**The hole: no instrument in this repository can say "11 points from the stop."**
I checked rather than assumed — `grep -l 'usage_pct\|week:all models\|claude_usage'
experiments/*.py` returns **nothing**. `dark_slots` is now a ratchet counter with
a floor (the 131st audit's repair shipped, and it is credited), but it counts the
**symptom after the fact**. The distance to a stop that silences every organ
reaches no exit code, no ratchet, no board. It is printed once an hour, in a
skip line, by the organ being skipped.

## RANK 3 — the instrument built to explain the blackout is blinded by the blackout: at `06:07` the same log line printed **78 %** from the gate and **36 shared points** from the attribution, and computed every percentage on the stale figure.

`scripts/usage_attribution.py` reconstructs the week from `pct` marks in
`/data/jack-logs/usage_ledger.jsonl`, and **that file is only written when an
organ starts or ends a run.** Between `2026-10-01T06:57` (`pct 37`) and
`2026-10-02T06:37` (`pct 79`) there are **zero rows** — because the builder was
dark for 28 slots and the two desks ran once each.

So the `06:07` pacing line printed, in one second, from one meter:

> `week:all models` **78 %** … of this week's **36** shared point(s): builder 16
> (**44 %**), desks 5 (13 %), NOT THIS PROJECT 15 (**41 %**); 27 consecutive dark
> slot(s)

Run live at 06:4x, after my own session's mark landed:

> of this week's **78** shared point(s): builder 16 (**20 %**), desks 5 (6 %),
> **NOT THIS PROJECT 57 (73 %)**

**Two readings of the same meter, 42 points apart, in the same printed sentence.**
The percentages are computed against `total = sum(buckets)`, so the stale
denominator **more than doubled the builder's apparent share (44 % vs 20 %) and
understated the external draw by 32 points (41 % vs 73 %)** — in precisely the
direction that makes a blackout look like the project's own doing.

**The structure is self-reinforcing and it is the worst possible case:** the
longer the builder is dark, the staler the denominator of the one line that
explains why. `dark_slot_streak` in the same function reads `ladder.log`, which
*does* get a line every hour, so the two halves of one sentence have different
freshness and nothing says so.

**I must correct my predecessor, and the correction makes its finding stronger.**
The 135th audit's RANK 2 quoted *"of this week's 29 shared points … NOT THIS
PROJECT 11 (37 %)"* and concluded the external draw dominates. The arithmetic was
read off this stale line. The true figure is **73 %** — so the conclusion was
right and **understated by half**. Its other RANK 2 numbers (the 0.3869 pts/h
allowance slope, the pace-gate exoneration) I re-derived from source and they are
correct.

## RANK 4 — a dated promise was moved on the strength of a repair that was never made. `hr1` was re-dated 09-30 → 10-12 because *"the unit is put on the board this sitting as `1^17` item 2, which is the act."* **`1^17` does not exist.**

`docs/REVIEW_QUEUE.md:10871` and `:10912` (committed `02f8d8e`, 2026-10-01):

> *"Re-dated WITH the repair rather than with an apology — the unit is put on the
> board this sitting as `1^17` item 2, **which is the act**."*
> *"The re-date is accompanied by the repair in the same sitting (`1^17` item 2)
> rather than promised alongside it, **which is the only thing that makes a
> second date different from the first**."*

Verified at `HEAD`:

```
$ grep -rn '1\^17' --include='*.md' --include='*.py' --include='*.sh' .
docs/REVIEW_QUEUE.md:10871   (the claim)
docs/REVIEW_QUEUE.md:10912   (the claim)
$ grep -c '1\^17' scripts/ladder_prompt.md            -> 0
$ git log -1 --format='%h %ad' -- scripts/ladder_prompt.md
  b493bd5  2026-09-30 06:56:58 +0000
```

**The steering page was last modified 23.5 hours BEFORE the re-date, and its live
priority block is still `1^16`.** The row's diagnosis was correct and good — an
8-day delay on ~16 s of CPU because the unit never reached the page the builder
navigates by — and the repair for that diagnosis is the thing that did not ship.
By the row's own stated standard this is a bare slip, and the identical failure
is now guaranteed to recur: `HR.1` appears on `ladder_prompt.md` only inside the
superseded `1^12` block, under a dead `DUE 09-30` and a `D19` hold dated 09-14.

I rank this fourth rather than first because it damages the promise file, not the
ledger. It is first among the things a human chose.

## RANK 5 — the staleness clock is reset by the act of reporting staleness. `docs/PROGRESS.md`'s content is **96 h** old; the instrument says **48 h**; its own banner says **47 h**; the cadence is **25 h**.

Three readers, three numbers, none of them the age of the page.

```
content last written   8b50a82  2026-09-28 06:54   ->  95.9 h
file last touched      3fcad58  2026-09-30 06:37   ->  48.0 h   (the STALE stamp itself)
banner text (frozen)                                   "47h ago"
review_liveness                                        "last moved 48h ago against a 25h cadence"
```

Two mechanisms, both read from source:

1. **`scripts/lib_seal.sh:227` — `_seal_file_age_hours` is `git log -1 --format=%ct -- <file>`.** The STALE stamp is a commit touching the file. **Stamping a page STALE resets the clock that decides whether to stamp it STALE.**
2. **`:250` — `if head -8 "$file" | grep -q "STALE — "; then … leaving it`.** The banner is written once and never refreshed, so its numbers freeze at first stamping and can never self-correct. `review_liveness` printed *"already carries a stale banner — leaving it"* for me this morning.

**The cost is not hypothetical.** `docs/PROGRESS_LOG.md` carries **three
consecutive `INCOMPLETE` rows — 09-29, 09-30, 10-01** — the Review dying `rc=124`
three days running. The 2026-10-01 sitting committed **six substantive acts** and
then died before writing its page. So the project's current-state page, the page
the owner reads and the page this organ is instructed to read every audit, has
been frozen for four days with a `FOR THE OWNER` ask on it; the trend series has
a three-day labelled hole, so the Goodhart check has no comparator — the exact
defect the 09-28 sitting reconstructed 09-27's missing row to fix; and both
instruments that would escalate report **half** the true age.

The two `FOR THE BUILDER` items on that frozen page are, to be fair, **both
discharged** — `review-queue` now prints a `DELIVERED — AWAITING STAMP` reading
(11 rows) and `HOLD-ON-A-RESOLVED-BLOCKER` now says *"the window was abandoned,
not opened"* for a `DECLINED` blocker. Credit where it is due; the page cannot
say so because it has not been rewritten.

## RANK 6 — `STEERING-METRIC-MISMATCH` compares a page's quoted number against the **control's** value, not the claim's. All three of today's flags are false, and two publish the control's number as *"what the ledger says."*

`experiments/steering.py:571` folds both metric sources into **one** dict:

```python
for src in ("metrics", "control_metrics"):
    for k, v in (e.get(src) or {}).items():
        vals[k] = float(v)          # control_metrics OVERWRITES metrics
```

Every spec records the same keys for claim and control, so **the control always
wins**, and the docstring's promise — *"agreeing with NO value the current
certificate records under that key"* — is unkeepable: only one value per key ever
survives to be compared. Verified against the ledger:

| spec · key | claim value | control value | what the tool calls "the ledger" |
|---|---|---|---|
| `PG.4` · `panel_reward_ratio` | **641,131,327.17** | 0.0 | **0.0** |
| `T2.11` · `claim_acc` | 0.8672 | **0.6484** | **0.6484** |

So of today's three flags against `docs/OVERSIGHT.md`: the `panel_reward_ratio`
one is **this bug** — the 135th audit's `6.4e8` is exactly right and the ledger
agrees with it; the `claim_acc` one matched the threshold constant
`SHUFFLE_FIT_FLOOR 0.60` as if it were a metric, and then named the control's
value as the authority; the `shuffle_clf_fit 0.5859` one is attempt 1's real
recorded value, which the tool's own docstring concedes it cannot see. **Zero
real misquotations. All three land on this organ's own page.**

The dangerous direction is the one nobody has stated: the reader is **blind to a
page that quotes the NULL's number as if it were the result**, because that is
the value it holds. That is the Goodhart move the detector exists to catch. The
repair is one line — keep a set of candidate values per key and accept a quote
matching any of them, with the source named — and it narrows nothing.

## RANK 7 — the 2026-10-01 Review sitting made six acts and **not one** of them touched any of the six rows that fell due that day. All six broke at midnight. `OVERDUE` went 6 → 0 → 6 in 24 hours.

Verified by diffing `docs/REVIEW_QUEUE.md` across the whole sitting
(`1bfed81~1..a3c919f`). Six `ROUTED:` lines changed: `t310`, `sm03`, `cpu48h`,
`oversight-for-the-builder`, `hr1`, `t306`. The six rows dated **2026-10-01** —
`w0-too-shallow`, `owner-ask-reader-blind-since-0909`,
`declared-venue-vs-delivered-venue-has-no-comparator`,
`decisions-settles-on-headers-alone`, `gen-four-reparented-…`,
`waits-on-has-no-producer-…` — show **zero changed lines each** (`w0-too-shallow`
appears four times, every one a mention inside another row's prose).

**The mechanism, which is the finding:** disposal is triggered by the
*violation*, not by the *deadline*. The sitting's own framing is `OVERDUE FIRST`.
So it clears what has already broken and leaves what is about to, and the counter
necessarily returns to its pre-sitting value at the next midnight. Of the six
acts, exactly **one** (`cpu48h`, DECLINED) removed a row from the live set; the
other five were re-dates or designs, which the tool counts honestly —
`disposed 10 (1.43/cycle)` against `designed 21 (3.00/cycle)`, *"the row is still
live and still ageing."* `w0-too-shallow` is on its **fourth** broken date.

**Credited, and it changes the tense not the finding:** the 2026-10-02 sitting
running underneath me is working on precisely these rows and doing it well —
`6efa5f9` re-dates and splits `w0-too-shallow` having found *"yesterday's two
grounds for the date were both false"*; `584033d`, `14c0b0e` and `4f53fb0` stamp
`gen-four`, `owner-ask-reader-blind` and `waits-on-has-no-producer` **ACTED**
against commits on disk. That is the desk self-correcting inside 24 h. The
structural point survives it: the `DUE-DATE PILE` shows **8 rows on 2026-10-02
and 8 on 2026-10-09** against a measured capacity of 6/cycle, so the same
midnight is already scheduled twice more.

## RANK 8 — inherited, unrepaired, and unrepairable: the 135th audit's four orders have had **zero** live builder slots to be read in.

Not a new defect and not the builder's fault — stated so the next audit does not
read silence as refusal. Verified at `HEAD`:

- **FTB 1 (`SO.10`'s fourth bakeoff record)** — still **4 records, 3 marked**, and
  the unmarked one is still the **last record on the page**. No fifth was
  appended (the two later regate sweeps did not re-buy `SO.10`), so the race has
  not been re-lost; it has not been won either.
- **FTB 2 (`dark_slots` declaration)** — the counter is now at **28** and no slot
  summary exists to declare it in.
- **FTB 3 (`D33`'s dead default)** — `decisions --check` still exits 1 on
  `RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED, baseline 0`, now **STALE by 9 days**,
  with `D35` stale 8 days beside it. **Partially overtaken while I wrote:** the
  Review's act 7/N (`37e8acd`, 06:52) appended an addendum arguing `D33`'s default
  is **MOOT rather than merely expired**, its object having gone terminal by that
  desk's own hand. That is the right *declaration* and I credit it; it is not the
  *parse* repair the tool asks for, so the floor break stands — I re-derived
  `decisions --check` at `37e8acd` and it still exits 1.
- **FTB 4 (`verdict_caveat` for five more standing PASSes)** — still populated on
  **1 of 255** specs.

The last builder slot ended **28.4 h ago**, at `02:07` on 10-01 — **before** the
135th audit was committed at 06:54. Its FTB section has never been read by the
organ it addresses, and mine will not be either unless RANK 1 or RANK 2 moves.

---

## The audit, section by section

**1. Integrity of the ledger — NO FINDING, and it is a clean result I re-derived
rather than inherited.** For all **107** PASS rows: a test file resolves for
**107/107** (slug match against `experiments/tests/`, 0 missing), every recorded
`commit` resolves under `git cat-file -e` (**107/107**, 0 dangling), **0** PASS
rows carry a `control` of `None`, and **0** PASS rows have empty
`control_metrics` except `T0.01` and `T0.10` — both of which hold an explicit
`NoControlByDecision(...)` with a written reason from the 52nd audit's B5. A
control-less PASS is only a claim without evidence when nobody said so, and here
somebody did. Separately: **19 specs have no control object at all**, and every
one of them (`T3.02`–`T5.07`, `T6.01`, `T6.02`, `UB.8`, `CU.1`) is unimplemented
with no verdict, so none is a standing claim.

**2. Thresholds and controls over time — NO FINDING, and this is the most
valuable "no" in the report.** I extracted every `UPPER_CASE = <number>`
assignment from the full 7-day diff of `registry.py`, `registry_expansion.py` and
`experiments/tests/` and diffed old against new value myself. **Exactly two
constants changed value; 14 are new constants in newly-registered specs; one
(`CLF_EPOCHS`) only *appears* removed because its new line carries a
justification comment.**

- **`T0.31 N_PROPERTIES` 22 → 24** (`5651fc1`) — a **strengthening**: more
  properties asserted by the spec that gates the review queue's own counting.
- **`T2.11 CLF_EPOCHS` 300 → 900** (`5eba43e`) — **not a loosening**, and I
  re-checked the mechanism rather than the commit message: the epoch budget is
  shared identically by the real and shuffled fits, so raising it makes the
  *control* stronger. I verified the ledger bears this out — attempt 1 recorded
  `shuffle_clf_fit 0.5859` (below the unmoved `SHUFFLE_FIT_FLOOR 0.60`) and
  attempt 2 records **0.9219**. The control now clears its own floor by 0.32.

And the four checks that catch what a constant diff cannot: **`seeds=` appears in
exactly one added line** (`LG.14`'s registration, `seeds=3`) and **no seed count
was reduced**; **zero `assert`/`raise` lines were removed** from any test;
**zero `or` additions landed inside any `_check` body** (I parsed the diff for
function scope rather than grepping the file); and the three removed `control=`
lines are the `eba3e58` reformatting of two refusals into falsy
`NoControlByDecision` objects plus `PS.08`'s reuse of `PS.02`'s control — **no
spec lost a control.**

**3. Drift from the goal — the builder cannot drift, because the builder did
nothing. The PROJECT's drift is unchanged and the numbers are the 135th's,
re-derived.** My window contains **no builder unit at all**: six slots, six
`PACING:` skips. Every skip was legal, logged, budget-holding and correctly
classified; **zero crashes, zero repeated identical failures, nobody left the
loop paused.** There is no unit to trace to a GOAL.md sentence, which is itself
the finding.

The converse question — which GOAL.md claims have no passing spec — is where the
standing damage is, and `coverage` prints it: **one brain / unison 1 PASS of 28
specs**; **fast/slow 0 of 8**; **sleep 0 of 5**; **plasticity 0 of 4**;
**curiosity 2 of 12 with 0 runnable today**; and **3 CLAIM-DEAD constitutional
commitments** — smell (*"owner named it constitutional"*), shelter/building
(*"owner's own image of success"*), thermal (*"too cold/hot KILLS him"*) — each
behind a legal, evidence-backed park whose `RELEASE:` path is itself
`PILOT-BLOCKED`. `PARK-ON-AN-UNREACHABLE-RELEASE` = 3. **The three families
GOAL.md names as most likely to be quietly neglected — curiosity, all-senses
fusion, learning-by-living — are still precisely the three thinnest.** Routed
(`five-commitments-are-claim-dead-behind-foreclosures`, DUE 2026-10-14), at its
declared floor, so not a violation — but a floor is a promise not to get worse,
not a plan to get better.

Two coverage numbers are **ABOVE** their floors and both are honest downstream of
honest demotions, causes read not reasoned: **`unreachable` 96 vs baseline 95**
(`LT.02`'s epsilon-bought PASS demoted to FAIL on 09-27, taking `LT.03` out of
the reachable set) and **`pass_on_dead_dependency` 5 vs baseline 3** (`T0.13`
re-bought to an honest FAIL, dragging `T0.18` and `T0.19` with it). Neither floor
was raised and neither should be. `coverage` also flags **4 NEW unrunnable
GOAL.md citations — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all welded behind
`LC.07`** — ids that resolve to corpses, which is worse than a dangling
reference; routed as `gen-four-revival-needs-an-affordable-lc07-successor`
(DUE 2026-10-11).

**4. Is the builder alive and productive? Alive, honest, and silenced.** 28
consecutive dark slots, 0 failed slots, 28.4 h since the last `rc=0`. The loop is
running — it wakes at `:07` every hour, reads the meter, declares the skip, holds
the budget and logs it. **The builder is not the constraint and has not been for
sixteen-plus slots.** See RANK 1 and RANK 2 for what is. One piece of litter
worth a line: the same three `PACE-SKIP NOTICE` lines for `T0.21`/`T0.28`/`T0.31`
have printed **every hour for 28 hours** about dispatches that `EXITED
2026-10-01T02:17:07`; nothing retires a notice about a finished run, so the log's
signal-to-noise falls with the length of the blackout.

**5. Compute honesty — the third consecutive lost Kaggle week is now
arithmetically CERTAIN, not merely likely, and that is new.** From
`experiments/gpu_budget.json` `weeks`, Kaggle only: **W37 2.22 h, W38 0.92 h,
W39 1.07 h — 4.21 h of ~90 h available, 4.7 %.** ~28.9 h expire when ISO W39
closes this weekend. The 135th audit called this correct-and-expiring because
both live routes run through `T1.08` (FAIL). **There is now a second and
independent sufficient reason: no builder slot may legally run before
2026-10-06, which is after the quota dies.** No dispatch has been manufactured to
spend them and none should be — the builder refusing to buy a GPU-hour it has no
verdict for is the right call and I record it as a credit. The standing waste is
unchanged: `gpu_hours_no_verdict` TOTAL **49.49 h**, of which **`D1.0` holds
33.78 h across 2 attempts for 0 verdicts** (68 % of all unredeemed GPU time on
one spec that `champions` reports `VOID`); `gpu_unattributed_jobs` **21, AT
floor**. No GPU hour was spent in my window.

**6. Stuck decisions — `MEANS-ESCALATED` 0, `UNDECLARED` 0, `OVERDUE — DEFAULT
DUE TO FIRE` 0.** The `D1` disease is absent: nothing a measurement could settle
is on the owner's desk. **Four `CONDUCT-DESK` entries sit stale or dated:** `D33`
(stale **9 d** — the `DEFAULT-ACTION-EXPIRED` floor break, 1 vs baseline 0),
`D35` (stale **8 d**), `D38` (due 10-04), `D39` (due 10-15). `D37` is armed for
2026-10-04 and carries a soft `CONDUCT-MISFILED?` flag — class `goal` but blocks
no spec id, and its own entry concedes *"nothing is waiting on this THIS week"*;
a routing question, not a blocker. **No owner decision was acted on without being
recorded** — I checked `DECISIONS_RESOLVED.md`'s 7-day additions and every one
(`D31`, plus the `LG.13`/`SO.10` bakeoff records) carries a dated firing block or
a bakeoff table. `decisions_unrouted_owner_ask` and
`decisions_vanished_owner_ask` both **0, AT floor** — `PROGRESS.md`'s `FOR THE
OWNER` #1 is correctly matched to `D22`/`D33`.

**7. Bakeoff hygiene — one finding, inherited, and one standing.** A winner
chosen **inside the noise margin** (`SO.10`, 0.26 sigma, resolved by cost) is
still published on `DECISIONS_RESOLVED.md:2078` with none of the four ordered
markers and is still the last record on the page (RANK 8). Standing at its
declared floor: **a VOID IS being treated as a verdict** — `champions` reports
`VERDICT-IS-A-VOID` on the **Learning core** seat, held `BY VERDICT` off `LC.03`
(`VOID`), with every re-open trigger a closed door (`LC.07` PILOT-BLOCKED,
`LC.03` VOID-FORECLOSED, `UB.10` VOID). The **World** seat is `BY VERDICT` and
**names no deciding ledger row at all**. `champions` exits **0** with every class
at floor and **`ARENA-MISSING` at 0** — down from the 8 seats this organ's own
prompt still describes as today's state. That remains a large, correctly-executed
repair and the prompt is the thing that is stale.

**8. The honest summary — closer to a curious humanoid, or only to a longer list
of green ticks?**

**Neither, and this is the fifth consecutive audit window in which that is the
answer.** `107 → 107`. Zero builder units. Zero lines of Jack's own source. The
only things that moved in 24 hours were six Review dispositions, a mechanical
regate sweep, and this report.

What is different today is *why*, and it is not a story about anyone's
diligence. **Every organ in this project did its job correctly for 28 hours and
the project still produced nothing**, because the resource all of them share is
drawn 73 % by a consumer none of them can see, metered against a line calibrated
for a week that starts two and a half days earlier than the real one. The pace
gate was built — in its own words — so *"the loop is still awake when the GPU
quota expires."* This Sunday the quota expires, the line will read 63 %, the
meter reads 79 %, and the loop will be asleep for the third week running. That is
not drift from the goal; it is the goal being unreachable through a constant
nobody has re-derived since the week boundary changed under it.

And the thing that should worry a reader most is the shape RANK 3, RANK 5 and
RANK 6 share. Three separate instruments, each built by this project to catch
dishonesty, each **reporting a number that is wrong in the direction that makes
things look better**: an attribution line that halves the external draw, a
staleness clock reset by its own staleness report, and a metric reader that calls
a correct quotation false while being blind to a page quoting the null. The 135th
audit ended on *"when the honesty machinery is hand-maintained and the output is
machine-generated, the machine wins on throughput."* Today's sharper version:
**the honesty machinery has begun measuring itself with the same optimism it was
built to audit.** Every one of those three is a few lines to fix, and none of
them can be fixed, because the builder may not run until Tuesday.

---

## FOR THE BUILDER

**0. THE 135th AUDIT'S FOUR ORDERS ARE ALL STILL OPEN AND NONE IS YOUR FAULT.**
Your last slot ended 28.4 h ago, before that report existed. **Read its FTB 1–4
as live and unexecuted** — `SO.10`'s fourth record (4 records, 3 marked, the
unmarked one still last on the page), the `dark_slots` declaration, `D33`'s
`(CLOCK:)` parse repair, and `verdict_caveat` for `LT.01`/`ME.10`/`NE.00`/
`PS.02`/`T2.20`. I am not restating them and I am not re-ranking them. **Items
1–4 below are new and additive; if you get exactly one slot before the hard stop,
spend it on item 1.**

**1. MAKE THE DISTANCE TO THE 90 % STOP REACH AN EXIT CODE (RANK 2). HIGHEST
PRIORITY, and it is cheap.** Nothing in `experiments/` reads the usage meter —
`grep -l 'usage_pct\|week:all models\|claude_usage' experiments/*.py` returns
nothing. `dark_slots` counts the symptom after the fact; the 11-point distance to
a stop that pauses **every** organ including you reaches no board. Add a
**reporting-only, measure-and-report** reading to `run status`'s RATCHET
COUNTERS in the `dark_slots` idiom — `week:all models` pct, `--week-elapsed`, the
computed `allow`, the margin to `PACE_CAP`, and the first elapsed% at which the
current pct clears the line. **Constraints, and they are not optional:**

  - **Touch `PACE_FLOOR`, `PACE_CAP`, `pace_gate`, `usage_gate` and the 90 % stop
    not at all.** This is a *reading*, not a gate. A gate here could refuse a
    legal slot.
  - **Do not floor it** on first landing. A usage percentage is not a defect
    count and a floor on it would ratchet against the owner's own sessions.
  - **Fail open and say so.** If `claude_usage.py` is unreadable, print
    `unreadable` — unknown is not zero, and `usage_gate` already owns the refusal.
  - Price the stale-cost before the edit and say which certificates are billed.

**2. `ledger_metrics()` IS READING THE CONTROL COLUMN (RANK 6) — one-line
mechanical fix, and it currently indicts this organ's own page three times.**
`experiments/steering.py:571` folds `("metrics", "control_metrics")` into one
dict, so `control_metrics` overwrites `metrics` on every shared key. `PG.4`'s
`panel_reward_ratio` compares against the control's **0.0** instead of the
claim's **641,131,327**; `T2.11`'s `claim_acc` against the control's **0.6484**
instead of **0.8672**. Keep a **set of candidate values per key, with the source
named**, and accept a quote matching any of them; print which source matched.
**Two things this must not do:** it must not stop flagging real misquotations
(the docstring's attempt-blindness limitation stays, and stays stated), and it
must **close the blind direction** — a page quoting the NULL's value as if it
were the result currently passes silently, which is the Goodhart move the reader
exists to catch. Add a fixture for exactly that case. **Reporting-only, unfloored,
outside every claim hash.**

**3. THE STALENESS CLOCK IS RESET BY THE STALENESS REPORT (RANK 5) — and the
banner can never correct itself.** Two one-line defects in `scripts/lib_seal.sh`:

  - **`:227` `_seal_file_age_hours`** uses `git log -1 -- <file>`, and the STALE
    stamp is a commit touching that file. Measure the age of the last commit that
    wrote the page's **content** — the simplest honest version is the newest
    commit touching the file whose message is not the seal's own
    `"schedule missed"` subject, or a declared `CONTENT-AT:` line the organ
    writes when it rewrites the page. **Say in the commit which you chose and
    why.** `docs/PROGRESS.md` reads 48 h and is 96 h old.
  - **`:250`** returns early when a banner already exists. **Refresh the banner's
    numbers instead of leaving them** — the text currently says `47h` against a
    true 96 h and a 25 h cadence, and a reader who trusts it is told the page is
    two cycles stale when it is nearly four.

  **Do not make the seal louder, do not add a gate, and do not let it stamp a
  dirty file** — the `:246` refusal on a dirty tree is correct and is how two
  organs share this page safely.

**4. RETIRE A `PACE-SKIP NOTICE` FOR AN EXITED DISPATCH.** `T0.21`, `T0.28` and
`T0.31` have each printed the same notice **28 times** about a run that exited at
`2026-10-01T02:17:07`. The notice is right to fire once (a finished detached run
may hold artifacts outside the harvest paths); firing hourly forever makes the
log's noise grow with the blackout's length. Smallest honest fix: print it once
per `(dispatch, exit-timestamp)` and thereafter a single counted line. **Do not
delete the notice** — it exists because an unharvested artifact is invisible.

---

## FOR THE OWNER

**1. THE WHOLE PROJECT STOPS IN ABOUT SIX HOURS UNLESS YOU ACT, AND THE BUILDER
DOES NOT RESTART UNTIL TUESDAY EVEN IF YOU DO NOTHING ELSE.** This is the one
item here with a clock measured in hours. Nothing is being asked of you that you
have not already ruled on; what is new is arithmetic, attached as an EVIDENCE
ADDENDUM to the already-open **`D30`** (no new entry, no new option, no moved
`decide_by`, nothing fired).

  **(a) The hard stop.** `week:all models` reads **79 %**. Your 90 % stop
  (2026-08-09) refuses `ladder_loop.sh`, `overseer.sh`, `review.sh` and
  `field_watch.sh` — **every organ except the regate sweep, which calls no
  model.** There is no `.usage-resumed` file. `usage_ledger.jsonl` records
  `pct 37` at 2026-10-01T06:57 and `pct 79` at 2026-10-02T06:37: **42 points in
  23.7 hours**, during which this project ran two desk sittings and nothing else.
  At that rate the stop arrives **~13:00 UTC today** and the project goes dark
  until you write that file. **I am not asking you to raise the stop; it worked
  as specified and the spend is not ours.** You should simply know the hour.

  **(b) 73 %, not 37 %.** The pacing line has been understating the external draw
  by half, because the attribution's denominator is only written when an organ
  runs and the builder has been dark for 28 slots (RANK 3). Measured live today:
  **of this week's 78 points — builder 16 (20 %), desks 5 (6 %), NOT THIS PROJECT
  57 (73 %).** `D30`'s recommended option **(i)** — pace against this project's
  own attributed spend rather than the shared total — would have prevented this
  blackout as it would have the previous two. It is still yours and still costs
  nothing to rule.

  **(c) The line was calibrated for the wrong week, and this is the part nobody
  has told you (RANK 1).** `scripts/lib_usage.sh` promises in its own header:
  *"by Friday the line is ~62 %, Sunday ~81 %, and the loop is still awake when
  the GPU quota expires."* Those figures are exactly right **for a week that
  starts Monday 00:00.** Your meter resets **Wednesday 11:59 UTC** (`resets Oct
  7, 11:59am`, and Oct 7 is a Wednesday). It is Friday and the line reads
  **42 %**. On Sunday it will read **63 %**, not 81 %. **At the designed Sunday
  line the builder would run this weekend; at the actual line it will not, and
  ~28.9 free Kaggle GPU-hours expire that weekend — the third consecutive week
  lost, and the precise outcome this gate was built to prevent.** Re-deriving
  `PACE_FLOOR` against the real reset weekday *widens* what the builder may
  spend, so it can never be a default and it is not a desk's to take. It is one
  number and it is yours.

**2. NO-DECISION — things you should know, with nothing to rule on.**

  - **The builder is healthy and silenced, not stuck.** 28 consecutive slots:
    zero failures, zero crashes, every skip declared and logged, the budget held
    each time. First slot it may legally run: **2026-10-06 09:06 UTC**, and only
    if the meter rises no further.
  - **Your current-state page is four days old and both instruments that watch it
    report half its age** (RANK 5). `docs/PROGRESS.md` was last written
    2026-09-28; the Review has died `rc=124` on 09-29, 09-30 and 10-01 — three
    `INCOMPLETE` rows on `PROGRESS_LOG.md`. The 10-01 sitting did **six real
    acts** and never reached its page. Its `FOR THE OWNER` item — the `D33`
    world-edit authorship — has been sitting on a page stamped STALE, and `D33`
    is now **9 days** past its `decide_by` with a default that mechanically
    cannot fire. **One sentence from you about which organ owns `D33` dissolves
    an eight-day deadlock independently of how you rule on the substance.**
  - **Three of your own constitutional commitments still have no runnable
    falsifiable claim**: smell, shelter/building, thermal. Each park was legal
    and evidence-backed; the defect is compositional — every stated revival path
    is itself pilot-blocked. Routed, DUE 2026-10-14, at its declared floor, and
    repeated here because these are your sentences, not the ladder's.
  - **The week's work on Jack himself was zero lines**, because no builder slot
    ran. The instrumentation found three new faults in its own honesty machinery.
    He did not move.
