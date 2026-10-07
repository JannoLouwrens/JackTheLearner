# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-07 06:37 UTC — 144th audit.**

## VERDICT: DRIFTING

**The ledger is clean and I want that said first, because it is the valuable
half and it is true.** All 107 standing PASS rows resolve to a spec in `BY_ID`,
every one of their `commit` hashes still exists in git, every one that declares
a control recorded `control_metrics`, and the two that declare none declare
`NoControlByDecision` with a written reason rather than having forgotten.
Section 2 is clean too: in seven days exactly two numeric constants MOVED, both
justified by a measurement in their own commit, and both in the honest
direction. Nothing was loosened.

**The drift is that three of this project's own readers are wrong, all three in
the direction that flatters somebody, and two of them are the readers the
project is using right now to argue a live owner decision and to steer the
builder.** The creature, meanwhile, has not been touched in a week: **203
commits in seven days, ZERO to `UnifiedBrain.py`, `TrainingPipeline.py`,
`playground.py` or `EpisodicMemory.py`** — the Review measured this on 10-04 at
249 commits and it is now one week deep, not a one-morning reading.

**Instruments, every one re-run immediately BEFORE this file was committed and
not quoted from the top of the sitting** (the 06:37 Review collision makes a
stale reading the default failure here): `coverage` **2**, `decisions --check`
**0**, `champions --check` **0**, `run status` **2**, `run review-queue` **2**.
Ratchet floors: **3 ABOVE** (`dark_slots` 21 vs 0, `pass_on_dead_dependency` 6
vs 3, `unreachable` 96 vs 95), 0 BELOW, 0 UNVERIFIED.

*My own conduct, disclosed: my first pass read `coverage` as 0 and `decisions`
as 0 by taking `head`'s exit status through a pipe instead of the tool's. I
caught it and re-ran. **`docs/LESSONS.md:8136` already carries this exact rule**
— "`set -o pipefail` is not the default in `sh`, and a `$(cmd | head)`…" — and
the 143rd audit disclosed the identical slip yesterday. Two consecutive audits
tripped a written lesson. I am not appending a second copy of it; the defect is
that the rule is in a file nobody re-reads mid-command, not that it is unwritten.
`decisions --check` genuinely exits **0** today, which is the register's first
green reading in 13 days and is NOT my pipe error.*

**Concurrency disclosed:** the Review DAILY started at 06:37:06 alongside me
and was still alive at 06:50 (pid 972129, 20 m timeout). It had already
committed six acts while I worked (`1b7a3ea`, `45fea3d`, `0554565`, `c43e3ff`,
`a8eb65b`, `34c3ba1`). I committed `docs/OVERSIGHT.md` and
`docs/DECISIONS_NEEDED.md` **by name**, and every instrument on this page was
re-read after those acts. Where this page quotes a queue number, the Review's
own later page is the authority.

---

## RANK 1 — the instrument that answers "who drew the meter that stopped the builder" is BLIND TO THE BLACKOUT ITSELF, and its bias is to blame the builder

This is the most damaging finding of the sitting because it is not an opinion:
I reproduced it deterministically off the real ledger, it is live in 20 lines of
`ladder.log` right now, and the number it corrupts is the evidence under an
owner decision that fires in three days (`D40`, `decide_by 2026-10-10`).

**THE MECHANISM, at source.** `scripts/usage_attribution.py:338-351` builds
`marks` from the `pct` readings in `usage_ledger.jsonl` and sums the deltas
between **consecutive recorded readings**. Rows are written only by an organ
starting or ending a run (`phase: "start"` / `"end"`). So while nothing runs,
no reading is written, and **the meter's rise during a blackout is not in the
ledger at all.** The instrument reports only up to the last mark — and
`pace_gate` prints it on the *skip* path, which is to say precisely when no
marks are being written.

**MEASURED, replayed, not argued.** The real ledger, through the pure
`attribution(text=…)` entry point, with nothing mutated:

| reading taken | total | builder | desks | NOT THIS PROJECT |
|---|---|---|---|---|
| at 06:07, 20 dark slots in (last mark 10-06T09:19, `pct 35`) | 27 pts | **20 (74 %)** | 4 (15 %) | **3 (11 %)** |
| at 06:37, the two desks' `start` rows land `pct 56` | 48 pts | **20 (42 %)** | 4 (8 %) | **24 (50 %)** |

Same week. Same builder spend — 20 points in both rows. The external share went
**11 % → 50 % purely because a reading got written.** The 21 points that fell
between 10-06T09:19 and 10-07T06:37 are **44 % of the entire week's rise** and
were invisible to every one of the twenty `PACING:` lines printed during the
blackout; all twenty print the identical `27 shared point(s): builder 20 (74%)`.

**WHY IT MATTERS AND WHO IS CURRENTLY MISLED.** `D26`'s fired default (iv) is
MEASURE ONLY — this file exists for no other purpose than to answer who drew
the pool. `D40` (armed, `decide_by 2026-10-10`, default (v) = status quo) is
being decided on these shares, and `D30`'s standing report quotes them every
sitting. The error direction is to **under-report the neighbouring tenant and
over-report the builder**, i.e. to make the gate's lockout look self-inflicted.
Today it inflated the builder's share by 32 points of percentage.

**WHAT I AM NOT CLAIMING.** The gate is not affected: `pace_gate` branches on
`pct` vs `allow` and never reads attribution. No certificate is touched. This is
a reporting defect — but it is reporting on which an owner ruling is three days
out.

**THE DESK THAT LOOKED AT THIS AND MISSED IT, re-derived rather than taken on
trust.** `builder-blackout-is-paced-by-another-projects-usage` (ACTED
2026-10-03) asked whether the channel can see the *desks*, proved it can
(`desks 0 → 5` untouched within one week), and closed noting one residual it
deliberately declined to route: that the channel "may under-count the desks —
but under-counting the desks blames the builder LESS, not more, so it cannot be
what is starving it." **That reasoning is correct and it is why this was missed:
the desk tested the one direction that was harmless and never asked about the
`unattributed` bucket, which fails the opposite way.** The row's own evidence —
a share moving without anyone touching the code — is this defect, read as
reassurance.

**THE REPAIR IS STRICTLY ADDITIVE AND I VERIFIED IT CANNOT LOOSEN ANYTHING**
(FOR THE BUILDER 1): `pace_gate` already holds a fresh `pct` *before* it
branches (`lib_usage.sh:80`). Have it append one mark row per slot. Checked at
source: `_rows` keeps any row with `ts` + `organ`; `marks` takes any row with an
int `pct`, so the mark is counted; `_sessions` pairs only `phase == "start"` /
`"end"`, so a `phase: "mark"` row creates **no** session and is credited to
nobody; `_week_rows` detects the reset by the meter falling, which more samples
make more accurate, not less. **Staleness bill ZERO** — `scripts/lib_usage.sh`
and `scripts/usage_attribution.py` are in **no** spec's `IMPL_DEPS` (census over
all `experiments/tests/*.py`).

---

## RANK 2 — the steering metric reader silently drops the CLAIM side of every certificate: 227 keys across 35 rows, and it is reporting MY OWN page wrong today

`run status` prints, right now:

    STEERING-METRIC-MISMATCH — 1 quoted metric(s) disagree with the live certificate.
      wall_s  docs/OVERSIGHT.md says 154.89 — ledger says W1.01 1149.54467

**The page is RIGHT and the reader is wrong, twice over.** The 143rd audit's
page says `wall_s` **1,154.89**. `W1.01`'s `metrics.wall_s` is
**1154.892643**. `_rounds_to("1154.89", 1154.892643)` returns **True** — I ran
it. The number the page quotes is exactly the number the certificate records.

**Three** independent defects in `experiments/steering.py` produce it, none of
them the two the open row prices:

**(C) `ledger_metrics` collapses `metrics` and `control_metrics` into ONE dict
keyed by metric name (`steering.py:627-651`), so on any key present in both,
the control value OVERWRITES the claim value.** The docstring one line above
promises the opposite — *"`metrics` and `control_metrics` both — pages quote
either"* — and `text_metric_mismatches` promises *"agreement with ANY of them is
silence."* The union is documented and not delivered. `W1.01` carries `wall_s`
in both; `control_metrics` wins; the claim-side value the page quoted is not in
the comparison set at all. **Scope measured ladder-wide: 227 metric keys across
35 certificates have a claim value differing from their control value, and for
every one of them the claim side is invisible to this reader.** It is both
noisier (flags correct quotations) and blinder (cannot check a claim-side number
at all) than its docstring says.

**(D) `_NUM = r"[-+−]?\d+\.\d+"` (`steering.py:567`) cannot read a
thousands-separated number.** On `1,154.89` it matches `154.89`. So the figure
the instrument attributes to the page is **not the figure on the page** — which
is why the printed row quotes a number that appears nowhere in `OVERSIGHT.md`.

**WHY THIS IS A ROUTED ROW'S THIRD AND FOURTH CAUSE, NOT A NEW ROW.**
`metric-reader-false-positives-were-60-percent-and-one-landed-on-the-audits-own-repair-order`
is OPEN, **`DUE: 2026-10-08` — tomorrow**. It prices exactly two surviving
causes: (A) the bar heuristic cannot see `<= RATIO_MAX 0.5`, and (B) the
agreeing reading sits outside the 60-char window. **Neither is today's row.**
The row declined to act because both its repairs "widen a reader's silence" and
a reader made quieter by its own implementer is the shape it exists to report.
**That objection does not reach (C) or (D):** both are correctness fixes that
change which number gets compared, not whether a comparison happens. Fixing (C)
makes the reader able to check 227 readings it currently cannot. Fixing (D)
makes it quote the page it is reading. **Staleness bill ZERO** —
`experiments/steering.py` is in no spec's `IMPL_DEPS`.

**(E) A THIRD CAUSE, FOUND BY THIS PAGE ITSELF AND STRONGER EVIDENCE THAN
ANYTHING I WROTE ABOVE.** Writing the paragraphs above made my own page a
specimen, and the reader now returns **two** rows against it. Their captured
values settle the argument without appeal to the docstring:

    wall_s  says 1154.89, 1154.892643, 154.89  — ledger says W1.01 1149.54467
    wall_s  says 1149.5446, 154.89             — ledger says W1.01 1149.54467

The first paragraph quotes **`1154.892643`** — byte-exact `W1.01`
`metrics.wall_s`, and `_rounds_to("1154.892643", 1154.892643)` is `True` — and
the reader reports it as disagreeing with the ledger. **That is cause (C) with
no interpretation required: the page quoted the certificate's own claim value to
full precision and the reader could not see it.** The second row exposes a
further defect: the page states `1149.54467` in full, and the reader captured
**`1149.5446`** — `_METRIC_WINDOW` is 60 characters and the number straddles the
edge, so it was **clipped mid-digit and then compared as a different number**
(`_rounds_to("1149.5446", 1149.54467)` is `False`, off by 0.00007 against a
0.00005 tolerance). A truncated capture manufactures a mismatch out of a page
that is exactly right. This is **not** cause (B) of the routed row — (B) is an
agreeing reading sitting *beyond* the window and being missed; (E) is a number
*inside* the window being misread. Disclosed rather than chased: fixing my
wording to dodge the reader would hide the specimen.

**And the reason this one stings:** the row's own text says a false positive on
the auditor's repair order is "worse than a silent one… the cheaper choice is to
stop reading the block." It has now landed on the auditor's page a second time,
by three mechanisms it has never named, on the morning before its own `DUE:`.

---

## RANK 3 — the Review's FTB 5 is the right repair with a premise that is FALSE AT SOURCE, and it says it verified it there

`docs/PROGRESS.md`'s `FOR THE BUILDER` item 5 orders the page-writing order
changed, and justifies it:

> *"The reason the ordering was adopted — not holding work dirty — **no longer
> applies**: `docs/PROGRESS.md` is a `PROSE_DOCS` member and exempt from the
> per-spec staleness bill, which this sitting verified at source."*

**It is not a `PROSE_DOCS` member.** `experiments/protocol.py:139-140`:

    PROSE_DOCS = ("CHECKLIST.md", "docs/LOOP_JOURNAL.md", "docs/LESSONS.md",
                  "docs/OVERSIGHT.md")

`docs/PROGRESS.md` is in **`INSTRUMENT_INPUT_DOCS`** (`protocol.py:186-190`),
and `protocol.py`'s own comment block explains at length why — the disposition
*did* intend the prose class and **measurement refused it**: *"`decisions.py:314`
binds `PROGRESS = _REPO / "docs" / "PROGRESS.md"` and `T0.28`'s import closure
reaches it, so classing it prose would have blinded a live reader on the first
day of the repair."*

**The bill the premise denies is being paid on this morning's board.** `T0.28`
declares `docs/PROGRESS.md` in `IMPL_DEPS`
(`experiments/tests/t0_28_decisions_tool_is_honest.py:127-128`), and `run
status` lists it under STALE CLAIMS **right now**:

    T0.28  recorded PASS; ... what moved: NOT its own test file ...; MOVED: docs/PROGRESS.md

`INSTRUMENT_INPUT_DOCS` is dirt *only for specs that declare it* — so the claim
is not merely wrong, it is **exactly inverted**: per-spec is the one place
`PROGRESS.md` does bill, and it bills a Tier-0 certificate every time the Review
rewrites the page, **in any order**. Reordering the page does not remove it.

**I am not asking for the order to be reverted, and I am not asking for a
reclassification.** The order is independently justified — **seven** consecutive
sittings have now lost their page (`PROGRESS_LOG.md`'s six `INCOMPLETE` rows
09-29→10-03, plus yesterday's `rc=124` at `0326fbb` and today's STALE stamp at
`1b7a3ea`), and the page on disk this morning carries *two* stacked banners
saying it is neither current nor complete. Moving `docs/PROGRESS.md` into
`PROSE_DOCS` is the wrong repair for the reason `protocol.py` already measured,
and it would cost 5 certificates to touch `protocol.py` besides
(`T0.15`/`T0.17`/`T0.27`/`T0.33`/`T0.35` declare it). **What must not happen is
the builder executing FTB 5 and recording "no staleness bill" on the strength of
a sentence that says it checked.** The order is sound; the warrant under it is
not; the bill is one Tier-0 re-buy per Review sitting and it should be stated.

---

## RANK 4 — the blackout, the arithmetic of when it ends, and the perishable cost nobody has multiplied out

Reported because `D30`'s fired default makes it standing, and because the two
halves have not been put in one sentence before.

**The builder is 21 consecutive dark slots deep; last `rc=0` 2026-10-06T09:19:21
(21.5 h).** `dark_slots` reads **21 against a declared floor of 0** — above
floor, growth nobody raised the constant for, and it may not be blessed by
`ratchets record`. One slot in the window died differently and is worth not
losing: `2026-10-06T15:07 ABORT: load 6.09 above 6.0` — the tenant-protection
refusal, correct behaviour, counted as dark.

**The cause this week is NOT the cause the pace line was built for, and I will
not recycle my predecessors' grievance.** `lib_usage.sh:47-53` justifies the
line by external drain — *"the loop is stopped by consumption it does not
control"* — and the 2026-09-09 Review proved that case with 38 of 61 points
accruing while nothing of ours ran. **This week the builder genuinely spent its
own 20 points in the eight productive hours of 10-06 and then paced itself out.**
The honest reading (RANK 1's corrected figures) is 48 points: builder 20 (42 %),
desks 4 (8 %), neighbour 24 (50 %). Both things are true and neither excuses the
other: the builder front-loaded, *and* half the week's meter is somebody else's.

**When it ends, from the line's own formula and not from a trend.**
`allow = PACE_FLOOR + ceil((PACE_CAP − PACE_FLOOR)·elapsed/100)` =
`25 + ceil(0.65·elapsed)`. Live now: `pct` **57**, `elapsed` **30 %**,
`allow` **45**. The builder runs again when `pct < allow`, i.e. when
`ceil(0.65·elapsed) > 32`, i.e. at **`elapsed ≥ 50 %`** — 20 points of the week
away, 0.20 × 168 h = **33.6 h**, so **≈ 2026-10-08 16:30 UTC**, and later if the
meter rises at all. The week resets ≈ 2026-10-12 04:30 UTC, which caps it.

**The cost, multiplied out, which is the part that is new.** `W40` (the budget's
Sunday-start week, Sun 10-04 → Sat 10-10 — verified at `gpu.py:409`, which
deliberately retired the ISO `%G-W%V` key, so the week label is right and the
one 10-05 job is correctly in W40) has drawn **0.271 h of 30 free Kaggle
GPU-hours. 29.73 h expire Saturday 2026-10-10.** The builder returns ≈10-08
16:30. **That leaves ~1.3 days of builder time to buy 29.73 h of free compute,
and the only designed buyer on the board is `T1.08` Step 1 at ~0.3 h.** This
would be the **fourth** consecutive week to expire near-empty: W37 5.249 h drawn,
W38 0.918, W39 1.072, W40 0.271 — **~112 free GPU-hours unbought in four weeks**
on a project whose owner ruled free compute only. Separately, `gpu_hours_no_verdict`
totals **49.77 h**, of which `D1.0` alone is **33.78 h across 2 attempts for 0
verdicts**.

Also dated into that lockout: at least five rows fall `DUE: 2026-10-08`
(`t215-router-under-lexical-null`, `lg12-abstention-knob-has-no-resolution`,
`t108-pipeline-repair-has-no-design`,
`longrun-binding-conjunct-went-false-four-times-under-an-unmoved-impl-sha`, and
the metric-reader row of RANK 2), and `d10-successor-rerun-under-adopted-gate` is
`DUE` **today**. Three of the five are `DISPOSITIONED` with an executing commit
already named and can be closed by a desk without the builder; the rest are dated
onto a day the builder provably cannot run, which this queue's own precedent
(`DUE: 2026-09-22` re-dates) calls knowingly manufacturing a violation. **Naming
it, not re-dating it — re-dating is the Review's act, not mine.**

---

## The audit, section by section

**1. Integrity of the ledger — NO FINDINGS, and I checked it myself rather than
reading `T0.18`'s verdict.** Over all **107** PASS rows: 107/107 resolve in
`BY_ID`; **0** have a `commit` that no longer exists in git (`git cat-file -e`
on every one); **0** declare a control and record an empty `control_metrics`;
**2** declare no control — `T0.01` and `T0.10` — and both carry
`NoControlByDecision` with the 52nd audit's written reason, which is a
declaration, not an omission. The known provenance debts are unchanged and
reported by the tool: 2 DIRTY STAMPS (`T6.03`, `PL.02`), 15 STALE CLAIMS, 6 PASS
rows predating `spec_sha`, 6 UNBACKED CERTIFICATES.

**2. Thresholds and controls over time — NO FINDINGS. Nothing was loosened.**
Full `git log -p` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` for 7 days. Every other `[A-Z_]+ = <number>` hit in the
diff is an ADDITION from a new spec (`T4.05`, `W1.04`, `LG.14`, `DP.04`'s
registration). Exactly two constants MOVED:

- **`CLF_EPOCHS` 300 → 900** (`t2_11_skills_distinguishable.py`, `5eba43e`).
  **Justified by measurement and it TIGHTENS.** Attempt 1 VOIDed because the
  *control* — `shuffle_clf_fit` 0.5859 < 0.60 — could not memorise 8 permuted
  labels: a rig-capacity failure, not a claim failure. Budget is shared by real
  and shuffled fits across all arms identically; the floor is byte-unmoved; the
  commit states held-out `claim_acc` likelier moves DOWN, so PASS gets harder;
  one knob turn only, with a second miss pre-committed to route rather than bump.
- **`_SEC_PER_SEED` 355.0 → 531.1** — a cost estimate revised UP on a measured
  redesigned rig. More honest, and it buys nothing.

Seeds: **no spec's seed count fell.** The only `seeds=` line touched reads
`seeds=3` on both sides (`15aa7a6`, T1.11's strengthening adding
`TrainingPipeline.py` to `IMPL_DEPS`). Assertions: `T1.11`'s `_check` has three
conjuncts at HEAD (`t1_11_path_parity.py:170-179`); both original conjuncts and
both original constants are byte-unmoved and the third is **conjoined with
`and`**, never substituted — I read the file, not the comment claiming it. No
`_check` gained an `or`. The one `_check` rewritten wholesale is `DP.04`'s, which
has **no ledger row at all** (never run) and was redesigned through the Review's
own `dp04` ruling with its bars pre-registered — a never-run spec's gate cannot
be a loosened certificate.

**3. Drift from the goal.** The last 24 h produced 21 commits. The builder's
share (01:07–09:19 on 10-06) traces cleanly: `T4.05` implemented — the forced
creature-gate unit completing `T6.01`'s harness chain, serving *"Unison (Tier
4)"* and *"A living Jack (Tier 6)"*; `W1.04` harvested to a first-ever VOID at
its pre-registered control lane, serving *"the world must be consistent…
consequential"*; `T1.12`'s disposition and the bare-commit race routed, serving
the honesty clause rather than a capability. None of it is drift. **The converse
question is where the answer is bad.** Still ZERO passing claim for **smell**,
**shelter/building** and **thermal ("too cold kills him")** — `claim_dead = 3`,
unchanged for eleven days, every claim spec PARKED or FORECLOSED, and
`coverage`'s PARK-ON-AN-UNREACHABLE-RELEASE shows all three revival paths
unwalkable today (`BA.02→LT.08` blocked, `SH.01→SH.02` and `SM.02→SM.03`
PILOT-BLOCKED). Of GOAL.md's named commitments, **14 have live claim specs with
nothing passing**, including `curiosity` (12 specs, 2 pass — both fixtures,
`LT.03` VOID), `one brain / unison` (28 specs, 1 pass), `fast/slow` (8 specs, 0
pass, five welded behind `LC.03`'s VOID), `sleep` (5, 0) and `plasticity` (4,
0). This is `D42` and it is correctly on the owner's desk; I am not re-asking.

**4. Builder alive and productive?** Alive, paced out, not broken. Last 24 h:
**4 iterations ran, 4 ended `rc=0`** (06:23, 07:39, 08:25, 09:19), then **20
`PACING:` skips and 1 `ABORT:` on load**. PASS delta **107 → 107** — zero, and
honestly zero: `W1.01` PASS and `W1.04` VOID are first-ever verdicts that landed
either side of the count, `T4.05` was implemented but is GPU-blocked behind
`T4.04`→`T1.08`. No thrash, no repeated identical failure, `failed_slots` 0. The
loop is not stuck; it is switched off by its own pace line (RANK 4).

**5. Compute honesty.** `experiments/gpu_budget.json`: W40 **0.271 h of 30**,
expiring Saturday 2026-10-10; the four-week run of near-empty weeks and the
49.77 h-without-verdict total are in RANK 4. The accounting itself I checked and
it is **sound** — I suspected the week label was wrong (today is ISO W41 and the
file's newest key is W40) and it is not: `gpu.py:409` records that the ISO
Monday-start key was deliberately retired because it "kept charging Sunday's runs
to" the wrong week, so the Sunday-start `%Y-W%U` label is correct and the 10-05
job belongs in W40. No overruns recorded. `gpu_unattributed_jobs` 21, AT floor.

**6. Stuck decisions.** `decisions --check` exits **0** — no `MEANS-ESCALATED`,
no `UNDECLARED` (so there is nothing for me to arm this audit), no
`OVERDUE — DEFAULT IS DUE TO FIRE`, 0 unrouted and 0 vanished owner-asks, and
31 of 31 firings transcribed. Two armed (`D41`, `D42`, both `decide_by`
2026-10-18); five `CONDUCT-DESK` of which `D33` (14 d), `D35` (13 d) and `D38`
(3 d) are stale. **I examined the one class that went green and it holds.**
`decisions_default_action_expired` fell **1 → 0** at `8ef62a5` by declaring
`D33`'s default date `(CLOCK: c9aca70)`. That is a shrink **by declaration**,
which is the mechanism `decisions.py:662-668` prescribes and the `D22` precedent
performed; the date was provenance of a re-date already executed before the
entry existed, nothing fired, no deadline moved, and `D33` still prints as STALE
by 14 days in the unarmed list, so the real defect is not hidden. **Legitimate.**
The thing I will say plainly anyway: `D33` is bucketed `CONDUCT-DESK` — *"execute
it, report it, do not ask"* — while the Review's own 10-02 ruling is that its
object went terminal and **no desk can clear it by firing anything**. A label
instructing execution on an entry established as unexecutable is a contradiction
the register carries silently. It costs nothing today and it will mislead the
next desk that reads the bucket and not the entry.

**7. Bakeoff hygiene — no findings.** 33 resolved entries; no decision recorded
without a learning gate, no VOID read as a verdict, no winner inside the noise
margin. The seat picture is `champions --check` **0** with all ten violations AT
their declared floors. The standing bad news is unchanged and I am not
re-litigating it: the Learning-core seat is held **BY VERDICT off a VOID**
(`LC.03`) with every re-open trigger a closed door, and the World seat names
neither a deciding run nor a rematch trigger.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder, or only to a longer list of green ticks?** Neither, this week, and that
is the least comfortable answer available. We are not closer to Jack: 203
commits and not one line of the brain, the body or the world. We are also not
closer to a longer list of ticks — `demonstrated` sat at 107 all week and the
Review *took one away* on purpose. What grew is the apparatus, and this morning
the apparatus is what I had to audit, because three of its readers are wrong:
one cannot see the blackout it exists to explain, one cannot see the claim side
of 35 certificates, and one page's order was changed on a premise its own source
file refutes in a comment. **That is still not failure.** A project where the
honest finding of the day is "our instruments mis-measured us in our own favour,
here is the replay" is a project whose ledger can be believed — and the ledger,
checked line by line today, can be. But the ladder-and-apple standard is a
creature that climbs, and the gap between the measuring and the measured is now
the widest it has been: the thing that is supposed to learn has been untouched
for a week, 29.73 free GPU-hours expire on Saturday with nothing designed to
spend them in time, and three of the owner's own constitutional commitments have
no living falsifier at all. We are getting very good at watching something that
is not moving.

---

## FOR THE BUILDER

1. **GIVE `pace_gate` A MARK, SO THE BLACKOUT IS ATTRIBUTABLE WHILE IT HAPPENS
(RANK 1).** In `scripts/lib_usage.sh`'s `pace_gate`, append one row to
`/data/jack-logs/usage_ledger.jsonl` per slot using the `pct` it **already
holds** before the branch (`:80`) — e.g.
`{"organ":"pacer","phase":"mark","pct":N,"ts":…}`. **Verify the defect yourself
first and stop and route if it does not reproduce:** replay
`usage_attribution.attribution(text=…)` against the real ledger truncated before
`2026-10-07T06:37` and you must get `total 27 / builder 20 (74%) /
unattributed 3 (11%)`, then untruncated `total 48 / builder 20 (42%) /
unattributed 24 (50%)`. Constraints I checked so you need not re-derive them:
the `organ` must be neither `"builder"` nor a member of `DESK_ORGANS`, and the
`phase` must be neither `"start"` nor `"end"` — otherwise `_sessions` invents a
session and `_alive` credits the span to somebody. **Strictly additive**: it adds
readings, touches no branch, moves no constant, and cannot change what
`pace_gate` returns. **Staleness bill 0** (no spec declares either file). Do
**not** change `PACE_FLOOR`, `PACE_CAP`, the line or the 90 % stop — none of this
is a licence to loosen the gate, and `D26` option (i) and `D40` option (ii) are
not yours.

2. **`experiments/steering.py` — TWO CORRECTNESS FIXES IN THE SAME READER, AND
THEY ARE NOT THE TWO THE OPEN ROW PRICES (RANK 2). The row is `DUE: 2026-10-08`,
so this perishes tomorrow.**
   - **(C)** `ledger_metrics` (`:627-651`) must keep BOTH sources' values per
     key, not overwrite — the union the docstring promises and
     `text_metric_mismatches`' "agreement with ANY of them is silence" rule
     requires. Measured consequence of the current flat dict: **227 keys across
     35 certificates** have a claim value that the reader cannot see.
   - **(D)** `_NUM` (`:567`) must accept thousands separators, so `1,154.89`
     reads as `1154.89` instead of `154.89`.
   - **(E)** a number clipped by the 60-char `_METRIC_WINDOW` edge must not be
     compared as if it were the whole number. Live on this page right now: it
     captured `1149.5446` from a page that states `1149.54467` and called it a
     disagreement. Widening the window is **not** the fix and would collide with
     cause (B), which the desk declined — the fix is to not emit a capture that
     the window truncated.
   
   **Verify all three yourself**: `_rounds_to("1154.892643", 1154.892643)` is
   already `True` and `metrics.wall_s` is already `1154.892643`, so after (C)
   both of today's `wall_s` rows must go silent — and they must go silent
   **because the values now agree**, not because a window or a heuristic got
   wider. Do **not** take causes (A) or (B) from that row; the desk declined
   them on purpose and the design of which readers get specs is the Review's
   (`seven-instrument-readers-are-gated-by-no-spec`, DUE 10-19). Report the live
   row count before and after. **Staleness bill 0.**

3. **DO NOT RECORD "NO STALENESS BILL" WHEN YOU EXECUTE THE REVIEW'S FTB 5
(RANK 3).** Its premise is false at source: `docs/PROGRESS.md` is **not** in
`PROSE_DOCS` (`protocol.py:139-140`), it is in `INSTRUMENT_INPUT_DOCS`
(`:186-190`), and `T0.28` declares it in `IMPL_DEPS`
(`t0_28_decisions_tool_is_honest.py:127-128`) — which is why `run status` lists
`T0.28` STALE on `MOVED: docs/PROGRESS.md` this morning. **Execute the reorder
anyway** — it is independently justified by seven consecutive lost pages — but
record the real bill: one Tier-0 re-buy per Review sitting, in any order. **Do
not** "fix" it by moving the file into `PROSE_DOCS`: `protocol.py`'s own comment
records that measurement already refused that (it would blind `decisions.py:314`,
a live reader), and touching `protocol.py` costs 5 standing certificates
(`T0.15`, `T0.17`, `T0.27`, `T0.33`, `T0.35`).

4. **The 135th–143rd audits' items and the Review's FTB 1–7 are still live and
none of it is your fault** — your last slot ended 21 slots before this line and
you are paced out until ≈2026-10-08 16:30 UTC (RANK 4). I am re-ranking nothing.
When you return, **`T1.08` Steps 0+1 keeps its rank above everything on this
page**: it is the only designed buyer of W40's 29.73 free GPU-hours, which expire
Saturday 2026-10-10, and items 1–3 here are all zero-GPU and keep. Item 2
perishes tomorrow and may ride in any slot without displacing `T1.08`.

---

## FOR THE OWNER

**1. `D40`'s evidence was wrong in the direction that argues against you
changing anything, and I have corrected it on the register rather than only
here. See `D40`.** It is armed with `decide_by 2026-10-10` and default (v) = the
status quo. The shares that entry and `D30`'s standing report are argued on come
from an instrument that cannot see a blackout while the blackout is happening
(RANK 1): the live line printed to the log twenty times during this outage said
the builder drew **74 %** of the week's meter; the same week, one reading later,
reads **42 %**, with **50 % not this project**. I have appended a
**`D40` — EVIDENCE ADDENDUM** to `docs/DECISIONS_NEEDED.md` in the idiom the
137th audit used for `D30`: it corrects numbers, proposes no option, moves no
`decide_by`, and fires nothing. **Nothing new is asked.**

**2. NO-DECISION: `D30`'s standing report, delivered here as its armed default
requires, with nothing to rule on.** Builder dark **21 consecutive slots /
21.5 h**, last `rc=0` 2026-10-06T09:19:21, against a trigger of 2× the hourly
cadence. By the pace line's own formula the builder returns at **`elapsed ≥
50 %`, ≈ 2026-10-08 16:30 UTC**, later if the meter rises; the week reset
≈2026-10-12 04:30 caps it. **`2026-W40` has 0.271 h drawn of 30 free Kaggle
GPU-hours and 29.73 h expire Saturday 2026-10-10** — the fourth consecutive
near-empty week (W37 5.249, W38 0.918, W39 1.072, W40 0.271; **~112 h unbought in
four weeks**). The only designed buyer is `T1.08` Step 1 at ~0.3 h, and the
builder gets ~1.3 days with it. All four organs fired within cadence: builder
hourly (06:07, a PACE-SKIP), overseer 6-hourly (now), Review daily (06:37 today,
still running as I write), field watch weekly. Silence is never success and
there is none. `demonstrated` **107, unchanged all week.**

**3. Nothing new is asked on `D41` or `D42`; this is a pointer.** Both are armed,
`decide_by 2026-10-18`, with monotone defaults and the Review's recommendations
quoted verbatim. I add one measurement to `D42`'s evidence and no opinion:
`claim_dead` has now read **3** for eleven days, and `coverage`'s
PARK-ON-AN-UNREACHABLE-RELEASE confirms that all three revival paths
(`SM.02→SM.03`, `SH.01→SH.02`, `BA.02→LT.08`) are unwalkable today, so the hole
cannot close by anything already registered. To `D41` I add that **203 commits in
the last seven days touched `UnifiedBrain.py`, `TrainingPipeline.py`,
`playground.py` and `EpisodicMemory.py` exactly zero times** — the Review's 10-04
reading of the same fact at 249 commits was not a one-morning artefact.

**4. NO-DECISION: what this audit did not do, named rather than omitted.** I did
not re-examine any PASS spec's design — that is the Review's Sunday jurisdiction
and the 143rd audit's page records eleven specs already nominated for it. I did
not price the `T0.28` re-buy that RANK 3 implies, because ordering a re-run to
produce a cleaner number is the one thing this organ may not do. And I arm no
decision this audit for the honest reason that there is nothing to arm:
`decisions --check` reports **0 `UNDECLARED`**.
