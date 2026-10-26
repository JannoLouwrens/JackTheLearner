# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-26 00:37–01:1x UTC — 162nd audit.** VERDICT: **DRIFTING.**

**Two disclosures first, because each changes how a number below should be read.**

1. **A regate sweep opened at 00:43:02 and is live in this tree as I write.**
   `git status` at my first read was ` M experiments/ledger.json` + ` M
   experiments/cpu_budget.json` — the sweep's own rotation work, not inherited
   damage (semantic check: `T0.17`/`T0.15`/`T0.22`/`T0.35`/`T0.36`/`SO.01` all
   `PASS a22`-class re-buys, `was PASS -> PASS`). **I wrote this page while it
   ran** — `docs/OVERSIGHT.md` is a `PROSE_DOC` and exempt from per-spec doc
   dirt — and I held every `docs/REVIEW_QUEUE.md` edit until the sweep ended,
   because `T0.21`/`T0.28`/`T0.31` declare that file and the open row
   `regate-dirty-guard-is-checked-once-at-entry-and-the-tree-went-dirty-mid-sweep`
   (`DUE 2026-11-01`) is exactly this hazard. Sweep-end and my own edit order
   are stated under THE BILL below.
2. **The 161st audit's two builder items are both discharged, verified at HEAD
   rather than from commit messages.** `cpu_budget.py`'s falsified
   single-process sentence was corrected at `434b7ed` and its priced bill paid
   (`T0.33`/`T0.34` both re-bought PASS at `a4b4d31`); `AT.01` and `WM.01` were
   both implemented and measured. Nothing of mine was left standing as a live
   order.

True exit codes, taken from the processes and not through a pipe, re-run after
my own edits: `coverage` **2** · `decisions --check` **0** ·
`champions --check` **0** · `run review-queue` **2** · `run status` **2** ·
`run ratchets` **2** · `run worst-seed-audit` **0**.

**One correction to my own working, disclosed because it is this project's own
named trap.** My first read of `coverage` was piped into `head`, so the
`echo $?` I took reported *`head`'s* status — 0 — and not the tool's. The true
code is **2** (`NO-LIVE-PATH`: the three CLAIM-DEAD commitments whose every
successor is PILOT-BLOCKED). `decisions`, `champions` and `review-queue` were
redirected to files and their codes were correct as first read. The lesson is
already on the board as *"never tail an instrument"*; the pipe form of it costs
an exit code rather than a finding, and it caught me once this sitting.

Ratchet slot line, re-derived before any edit of mine:

> `8 MOVED (fail_unowned_owned_forms {'disposed': 1, 'queue-row': 27,
> 'repaired_by': 1} -> {'disposed': 2, 'queue-row': 28, 'repaired_by': 1},
> gpu_hours_no_verdict [map, TOTAL 51.47 h -> 53.24 h],
> pass_on_dead_dependency 6 -> 7, projection_vs_estimate [+1 T2.05],
> review_queue_net_arrivals -5 -> 6, review_queue_piled_on 15 -> 5,
> review_queue_violation_forms {'OVERDUE': 16} -> {'OVERDUE': 10},
> review_queue_violations 16 -> 10); 1 day-rolled (cpu_foreclosed_now); no
> counter refused to compute; floors: 2 ABOVE (pass_on_dead_dependency,
> unreachable), 0 BELOW, 0 UNVERIFIED.`

`claim_dead = 3` · `commitments_uncovered = 0` · `fail_unowned = 6` (at floor) ·
`champions_trigger_debt = 3` · `champions_unwinnable = 4` (at floor) ·
`goal_unrunnable = 7` · `dark_slots = 0` · `unreachable = 96` (floor 95) ·
`pass_on_dead_dependency = 7` (floor **3**) — and the last one is RANK 2, which
is the first page in this series to name its seven members.

**Why this is DRIFTING and not INTEGRITY RISK, stated rather than hedged.** Two
populations of standing certificates below have foundations nobody can currently
examine, and one confirmed false PASS entered the ledger yesterday. But the
false PASS was caught by its own author in eleven minutes, superseded by a
strictly harder gate, and left in `history` where I could read it; both
populations are *declared by this project's own counters*; all 108 PASS rows
check out at source; and section 2 found no loosening. That is a project whose
instruments are telling the truth about weak foundations, not one whose ledger
is lying. The distance to INTEGRITY RISK is one unread instrument — which is
exactly RANK 1.

---

## RANK 1 — the class that wrote a false PASS yesterday is detectable only AFTER the false PASS exists, by a tool nothing in this project runs, and the ruling that would fix it falls due TODAY onto a pile

**This is the most serious class on this page because it is the only one that
can put a wrong verdict in the ledger while every gate stays green — and it
did, 5 hours before this audit opened.**

### What happened, off the record rather than argued

`WM.01` recorded **PASS, attempt 1, at 2026-10-25T19:26:45** (`e3560bc`). Its
deciding conjunct was `m["buffer_vs_no_buffer_margin"] >= MARGIN_MIN` (0.10).
`run_spec` aggregates per-seed metrics by MEAN and calls `_check` once, so that
conjunct read the **mean 0.1927** while the per-seed margins were
**{0.4707, 0.0800, 0.0273}** — two of three seeds below the bar the spec's own
docstring had pre-registered as *"on every seed"*. The builder found it by hand,
shipped a worst-seed boolean gate (`b68d9e2`, bars byte-unmoved, strengthen-only),
re-ran, and `WM.01` is now an honest **FAIL**. The superseded PASS is still in
the row's `history` — disclosed, not buried. **Credit where it is owed: that is
the organ catching itself, and it is the best conduct on this page.**

### The instrument that exists for this class could not have prevented it

`aggregate-hides-worst-seed` was ruled on 2026-09-30 **(c) THEN (b)**, and arm
(c) — `experiments/worst_seed_audit.py` + `run worst-seed-audit` — shipped
2026-10-01 (`5b026bd`). Read at source this sitting, `worst_seed_audit.py:521`:

```python
if g.key in folds:
    fold_rows.append(entry)
```

**LANE A fires only when the gated key is produced by a `min`/`max`/`len` fold
in the same file.** `buffer_vs_no_buffer_margin` is a bare subtraction. So the
lane built to catch this pre-verdict was **structurally blind to it**, and the
blindness is not a bug — it is the ruling's own specification of arm (c)
(*"flags a bare `m["<key>"]` comparison on a key the same file folds with
min/max/len"*).

**LANE B — the lane that WOULD have caught it — reads the committed record.**
Its bound is `mu ± sigma*sqrt(2)` (the exact extreme-deviation bound for n=3).
On `WM.01` attempt 1: `mu` 0.1927, `sigma` 0.1978, so `mu − bound` = **−0.087**,
on the failing side of `>= 0.10`. It would have fired — **but only after the
false PASS was already in `ledger.json`**, because a row with a `std` has to
exist first. For a newly written spec this class is therefore *undetectable
before the verdict and detectable only after it*, and nothing in this repo says
so.

### And nothing runs it

Checked at source, not assumed:

| channel | reads `worst-seed-audit`? |
|---|---|
| `crontab -l` (`ladder_loop.sh`, `review.sh`, `regate.sh`, `field_watch.sh`, `overseer.sh`) | **no** |
| `scripts/*.sh` | **no** |
| `run status` | **no** |
| `scripts/ladder_prompt.md` | names the ROW (`:894`, `:907`); **orders no run** |
| any ratchet counter | **none exists** |
| exit code | **0, always** |

Last time its numbers reached a committed document: **2026-10-01** (`3204870`,
`e56193a`). Twenty-five days.

### What it says today, and what that costs

Live reading: **LANE B — 17 gates across 13 specs whose committed record admits
a violating seed.** Cross-referenced against status by me, which no page has
done before — **six are standing PASS**:

| spec | gate | record admits a seed at | bar |
|---|---|---|---|
| `LT.01` | `m["adv_ge_bar"] > 0.0` | −3.44e-07 | `> 0` |
| `ME.10` | `m["skill_gain"] >= MIN_SKILL_GAIN` | 0.236832 | `>= 0.25` |
| `NE.00` | `m["cycle_best_discounted"] < 0.0` | 0.00702 | `< 0` |
| `PG.4` | 4 gates incl. `icm_dwell_share >= 0.4` | −3.44e-07 | `>= 0.4` |
| `PS.02` | `m["shuffled_r2"] <= SHUFFLED_R2_MAX` | 0.0999 | `<= 0.05` |
| `T2.20` | `c["search_time_ratio_recency"] >= CONTROL_RATIO_MIN` | 0.57023 | `>= 0.75` |

Arithmetic spot-checked by hand on two of them (`ME.10`: 0.370367 −
0.0944234·√2 = 0.23684 ✓; `T2.20`: 0.992033 − 0.29826·√2 = 0.57024 ✓) — **the
tool's arithmetic is right.**

**Stated precisely so this is not read as bigger than it is: "admits" means the
aggregate cannot EXCLUDE a violating seed, not that one violated.** For a 3-seed
mean and std, the extreme deviation is bounded by `sigma*√2`, and these six sit
inside that bound on the wrong side of their bar. It is an *unresolvable* record,
not a refuted one. **Of the six, exactly one — `PG.4` — carries a board-visible
caveat** (`run status` prints its `{1,1,0}` disclosure, 90th audit B3). The other
five are green ticks with nothing beside them.

### The ruling that fixes it is due TODAY, and the date it was given is no longer empty

Arm **(b)** — the recorder refusal, which the 09-30 ruling itself calls *"the
only arm that makes the bug unrepeatable"* — is **still unordered, 26 days after
its price was measured**. STEP 2 is the **desk's**, `DUE: 2026-10-26`. The row's
own words choosing that date:

> *"2026-10-26, not the instrument's `next date with room` (2026-10-23) … this
> is a design ruling that needs a sitting's attention rather than a slot in a
> batch. **10-26 carries nothing.**"*

**10-26 now carries SEVEN live rows and `review-queue` prints it `!! AMBER:
pile`** against a measured capacity of 6 — and the Review DAILY that opens at
06:37 inherits **10 OVERDUE rows** which `D28`'s armed default puts *first*.
The 10-14 date for this row broke for precisely that reason, recorded on the row
("*it broke because no desk sat between 10-14 and today*"). I am reporting it
**before** the date passes, which is the only time it is actionable.

### And the price STEP 2 must rule on is stale

The row quotes arm (b)'s breakage price as **LANE A 139 fold-flagged pairs**.
**Live it reads 148** — +9 gating sites across 63 specs. The desk is scheduled to
rule today on a number nine sites out of date. This is a live worked example of
the open row `register-entries-quote-instrument-readings-that-nothing-re-derives`
(`DUE 2026-11-02`) landing on a ruling due the same week.

**ROUTED** as `worst-seed-lane-a-is-blind-pre-verdict-and-lane-b-reads-only-the-damage`
(`DUE: 2026-11-06`), with the `WM.01` worked example, the `:521` source line, the
six standing-PASS cross-reference and the 139→148 delta on the row. It is
explicitly **NOT** a re-route of `aggregate-hides-worst-seed` — checked at
source: that row's subject is *whether the recorder should refuse*, this one's is
*that its own measuring arm cannot speak until a verdict is already wrong, and
nothing invokes it either way*. The row **refuses in advance** any repair that
lowers a worst-seed count by narrowing the audit's domain.

## RANK 2 — the seven least-trustworthy PASS certificates in the ledger are exactly the seven the only re-buy lane is forbidden to re-measure, and no page has ever named them

`pass_on_dead_dependency` reads **7 against a declared floor of 3** — the worst
floor breach on the board, and the 161st audit reported the count. **It has never
been enumerated.** Here it is, computed live from `coverage.pass_on_dead_dependency`:

| PASS certificate | rests on | that dependency's status | last ACTUALLY ran | age |
|---|---|---|---|---|
| `T0.19` | `T0.13` | FAIL | 2026-08-11T20:37 | **76 d** |
| `T2.03` | `T1.08` | FAIL | 2026-08-19T20:32 | **68 d** |
| `T0.18` | `T0.13` | FAIL | 2026-08-30T08:18 | 57 d |
| `T2.14` | `T1.08` | FAIL | 2026-08-30T07:09 | 57 d |
| `LF.02` | `T6.03` | BLOCKED | 2026-09-06T12:23 | 50 d |
| `T1.12` | `T1.11` | FAIL | 2026-09-26T23:47 | 30 d |
| `PL.00` | `T0.07` | FAIL | 2026-10-21T04:45 | 5 d |

**The compounding fact, read off this sweep's own log minutes ago.** The regate
rotation is the only organ in this project that routinely re-buys a standing
certificate, and `/data/jack-logs/regate.log` at 00:43:04 today:

```
skip T0.18   rotation would record BLOCKED, not a verdict — dependency red: T0.13 (FAIL)
skip T0.19   rotation would record BLOCKED, not a verdict — dependency red: T0.13 (FAIL)
skip PL.00   rotation would record BLOCKED, not a verdict — dependency red: T0.07 (FAIL)
skip LF.02   rotation would record BLOCKED, not a verdict — dependency red: T6.03 (BLOCKED)
```

Counted over the whole log: `T0.18`/`T0.19`/`LF.02` have been skipped **96
consecutive sweeps** (since 2026-10-17T16:43), `PL.00` **34** (since
2026-10-23T06:43). The remaining three are outside the lane for unrelated
reasons — `T1.12` is refused as `cpu<10min` against a measured 1141 s, and
`T2.03`/`T2.14` are GPU. **So all seven are exempt from re-measurement, and four
of them by a rule written for their protection.**

**The skip is CORRECT and I am not asking for it to change.** Rotating them would
record `BLOCKED` — destroying a PASS by bookkeeping rather than by measurement,
which is strictly worse, and the log line even names the owner (*"owned by its
pass_on_dead_dependency row"*). **The defect is the pair, and that nothing prints
it:** the mechanism that shields these certificates from a bookkeeping demotion
is the same mechanism that exempts them from ever being falsified again, and
`pass_on_dead_dependency` counts the shielding without counting the exemption.

**Three of the seven hold champion seats**, which is how this reaches the
architecture invariant: `T2.03` is the **Vision encoder** seat's arena (held BY
DEFAULT, and `champions --check` already prints that seat `UNCONTESTED`);
`PL.00` is **Audio encoder** *and* the **PLASTIC-ONLY decree** seat (also
`UNCONTESTED`); `LF.02` is **Death & persistence**'s only PASS arena member and
the file's single `KINDLESS DISCHARGE`. Two instruments each hold half of this
and neither can see the other's half.

**The floor is unreachable by construction, and that is the part the owner should
see.** The only honest way down is the dependency going green: `T0.13` FAIL,
`T1.08` FAIL *with a sixth attempt prohibited in writing*, `T0.07` FAIL (routed
1 d ago), `T1.11` FAIL (`D41` fired (iii) — no call site added), `T6.03`
BLOCKED behind `T2.10` FAIL. **Five of five blocking dependencies are red, four
with no reachable repair.** So `7 vs 3` is not a backlog that drains; it is a
standing state with no path, and I may not propose raising the floor.

**ROUTED** as `pass-on-dead-dependency-certificates-are-exempt-from-the-only-rebuy-lane`
(`DUE: 2026-11-06`). Distinguished at source from two neighbours rather than
assumed distinct: `pass-certificates-are-not-re-evaluated-when-a-dependency-falls`
is `ACTED` 2026-09-22 and terminal (its subject was re-evaluation ON the fall, not
permanent exemption after it), and `staleness-of-a-standing-pass-reaches-no-exit-code`
(OPEN, `DUE 2026-10-27`) is about staleness-by-content, not dependency death. The
row **refuses in advance** the repair that would make the number fall — letting
the rotation record `BLOCKED` on these seven.

## RANK 3 — a tenant-memory breach happened last night, and the one organ that could have reported it is structurally prevented from seeing it before its own report is written

`SYSTEM.md` and `CLAUDE.md` both rank tenant safety above the ladder. Last night,
`2026-10-25T21:23:51`, `ladder.log`:

```
MEMORY 3519175:1589127631 — peak rss 1736 MB (VmHWM) over the 1536 MB ceiling,
  305s CPU, cmd: ... lg_15_character_sheet_answers_for_no_one import llm_pass
```

**113% of the ceiling, on a box with four paying tenant agents.** `D18`'s armed
default is *MEASURE AND REPORT, GATE NOTHING, RELAX NOTHING*, so naming it is the
whole obligation — and the naming reached exactly one place.

**Why the builder did not report it, and this is mechanism rather than blame.**
The 21:07 slot's committed journal entry (`LOOP_JOURNAL.md:24927`) is a full,
careful record of `LG.15` and does not mention the breach — because the loop
emits the `MEMORY` line at **iteration end (21:23:51)**, after the slot summary
has been composed. The organ that caused the breach cannot, by the loop's own
ordering, carry it into the durable record.

**Why no row holds it either.** The breaching process was a bare
`python -c ... llm_pass()` under `launch_detached.sh`. `run_spec` writes
`peak_rss_mb` into the ledger row and a receipt to `/data/jack-logs/mem_receipts`
— **this process called neither.** `LG.15`'s own ledger row carries the *harvest*
process's footprint, which only reads an artifact. Verified mechanically: of 164
ledger rows, **10 carry `peak_rss_mb >= 1536`** (`T1.03` 7389.1, `T0.07` 6897.5,
`T0.04` 3407.9, `T0.16` 2737.0, `LC.02` 2132.7, `T1.04` 2073.9, `PG.6` 2073.4,
`T0.14` 1895.6, `T1.12` 1668.6, `SO.01` 1592.1) and **1736 MB is not among them.**

**And nothing reads the population.** `grep` over `experiments/` and `scripts/`
for `peak_rss_mb` outside tests returns only `protocol.py` WRITING it
(`:637`, `:3739`, `:4248`, `:4417`, `:4429`). `D18`'s firing record states the
population as *"8 of 69 (12%)"*; live it is **10 of 164**. Nobody re-derived
that in the 54 days since, because nothing can.

**This is the third instance of one shape in two days** and that is why I am
ranking it rather than filing it as housekeeping: the 161st audit's RANK 2 found
both recorders of a lost builder slot erase themselves on recovery; `LESSONS.md`
`495dd4b` recorded that `cpu_budget`'s projection was proved safe in the
direction that costs a dispatch and unexamined in the direction that costs the
box. **Every counter in this project that would show it doing harm either resets
when the harm ends or was never built.** Here the harm is to the tenants, which
outranks everything else on this page by SYSTEM.md's own ordering.

**ROUTED** as `detached-lane-memory-breach-reaches-no-row-no-receipt-and-no-counter`
(`DUE: 2026-11-06`), constrained on the row to the REPORT half only:
`JACK_MEM_CEILING_MB` 1536 is byte-unmoved, `~1.5 GB` in `SYSTEM.md` is
byte-unmoved, nothing is gated and nothing is killed — `D18` fired *GATE
NOTHING* and this organ may not propose otherwise.

## RANK 4 — OVERDUE went 18 → 0 yesterday and was 10 again by midnight, from rows the desk had in front of it — reported as a measured instance, deliberately NOT re-routed

Yesterday's Sunday FULL page claims `queue | **OVERDUE 18 → 0**`. True when
written. **Six hours later it is 10**, and every one of the ten reads
`promised 2026-10-25 (1 d ago)`:
`two-eyes-one-certified`, `t205-world-model-loses-to-the-ridge-reference`,
`freeze-release-condition-is-five-specs-deep…`,
`taskmanager-decompose-documents-a-learned-path-with-no-call-site`,
`seven-rows-are-held-behind-a-refused-window-and-a-moot-decision`,
`t111-certified-loss-has-no-shipped-caller`,
`xl00-negative-claim-does-not-assert-its-own-test-could-reject`,
`t112-pass-stands-on-t111s-orthogonal-added-conjunct`,
`longrun-population-grew-six-ids-under-a-priced-row`,
`hold-on-a-resolved-blocker-did-not-fire-on-a-live-row-that-declared-one`.

**The desk was not blind to it:** `review-queue` printed `IMMINENT — 17 live
dated rows fall due on or before 2026-10-26 … 11 of them cannot be discharged`
in the same run that printed `OVERDUE 0`. Ten promises were hours from breaking
while the headline read clean. **This is the open row
`overdue-reconstitutes-overnight-from-rows-the-desk-could-see` (`DUE 2026-10-31`,
routed 4 d ago) promoted from hypothesis to a ten-row measured instance.** I am
NOT re-routing it — that row owns it, and re-asking is the churn
`VANISHED-OWNER-ASK` exists to catch. Recorded here so the instance is on a
page, and because it is the mechanism that will crowd out RANK 1's STEP 2 at
06:37.

Beside it, honestly: the Sunday FULL that wrote the page **died `rc=124` at
13:02** (40 min wall, started 12:22) and sealed `PROGRESS.md` as a draft. Its
own `Part 2` declares *"this sitting did NOT complete the 8–12 spec sample"*
because the OVERDUE class consumed the first half. The 161st audit read both
seal banners and judged them *"honest about the exit code and inert as a
steering control"* because the page carries its closing receipt — I re-derived
that and agree; the page is complete in content. It is the **work** that was
truncated, and the page says so itself.

---

## Section-by-section, including the clean results

**1. Integrity of the ledger — CLEAN.** All **108** PASS rows checked
mechanically: every declared implementation file exists, **every recorded
`commit` resolves in git (0 dangling)**, every spec declares a `control`. The
two that read controlless are `T0.01` and `T0.10`, both carrying explicit
`NoControlByDecision` objects (52nd audit), verified by type. The ledger holds
**164 rows** against 261 registered specs: PASS 108, FAIL 37, VOID 18,
BLOCKED 1. The caveat is not in the rows — it is RANK 1 and RANK 2, which are
about what 13 of those 108 certificates are *standing on*.

One thing checked and found CLEAN that I expected to find dirty. `LG.15` measured
last night that the shipped `ResponseGenerator.answer_question` **answers nobody
at either end** (sheet present: the 670-token persona truncates the question away
at `max_length=256`; sheet absent: the first sampled token is `<|im_end|>`
because the raw-completion prompt never applies the chat template — all 30
generations empty). The obvious worry is that `LG.00` — which `GOAL.md` cites by
id as *"the proof he is a creature and not a costume"* and which seats two
champion seats — bought its PASS on a path that produces nothing, i.e. that its
required COLLAPSE was the empty-string artifact. **It did not.** Read at source:
`LG.00` scores candidate answers as *continuations* by length-normalised
log-probability and takes the argmax (`LG.01`'s readout, by import) — it never
free-generates, so the defect cannot reach it. `LG.00`'s PASS stands. The real
cost is a different one, already owned by three open rows
(`certified-diary-is-not-the-memory-the-shipped-brain-carries`,
`t111-certified-loss-has-no-shipped-caller`,
`shipped-companion-cannot-move-see-or-report-failure`): the certified claim runs
on a scoring harness the shipped brain does not use, and the shipped brain's own
answering path is dead. Not re-routed.

**2. Thresholds and controls over time — NO LOOSENING FOUND.** `git log -p
--since="7 days ago"` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` is 5,143 lines across 21 commits. Five touch gates and all
five move in the strengthening or neutral direction:

- `b68d9e2` (`WM.01` worst-seed gate) — read line by line. `MARGIN_MIN` 0.10,
  `LEARN_MIN` 0.25, `CONTROL_MAX` 0.30 all **byte-unmoved**; `_check` went from
  two bare mean comparisons to two per-seed booleans gated `== 1.0`. It can only
  turn PASS into FAIL. The incident that forced it is written into the docstring
  with the per-seed numbers, not into a commit message that scrolls away. **This
  is the correct handling of a self-caught false PASS and I want it on the
  record as such.**
- `cea61ef` (`AT.01` pre-registration) — per-seed booleans **from authorship**,
  so `WM.01`'s defect cannot recur there; both null-only pre-check readings in
  the docstring; two must-fail controls, both held per seed.
- `f87ca40` / `0b5a85a` (`LG.15` registration + implementation) — all three
  ruling conjuncts binding, none weakened; the provenance conjunct is a VOID
  lane, and it fired, which is a control working.
- `4032e9c` (`LC.08` declared limit) — a comment, no constant.
- `dfb71d1` (`t306` probe V2) — a boundary epsilon replaced by
  round-to-recorded-precision exact compare: strictly tighter.

Live spot-checks rather than inherited: `MAX_POLICY_REL_STD` reads **0.20** at
HEAD (the constant `T0.07`'s attempt-7 FAIL was measured against, unmoved since);
`RANDOM_DWELL_MAX` **0.0185**.

**3. Drift from the goal — the day's work is on-goal and the ratio is the best
in weeks; the converse has not moved.** Seven builder slots since 18:07, **all
seven `rc=0`**, PASS delta **+1** (107 → 108) with the registry growing 260 → 261:

| slot | unit | which `GOAL.md` sentence |
|---|---|---|
| 19:1x | `cpu_budget` comment + bill; **`WM.01` FAIL** | *"fast and slow … must be in the research and tests"* — the `fast/slow` commitment's **first-ever measurement** |
| 20:1x | **`AT.01` PASS**; `LG.15` registered | *"the sense that works when sight fails"* — reallocation when vision goes dark |
| 21:1x | `LG.15` implemented + dispatched | *"develop knowledge and connect it to words instead of just the LLM … pretending to be Jack"* |
| 22:1x | **`LG.15` VOID** harvested, venue dead at both ends | same — and the VOID is the finding |
| 23:1x–00:1x | `DP.04` option (ii) knob probe, **no-candidate branch fired** | *"the world must be consistent, discoverable, consequential"* |

**Nothing in the last 24 h is drift.** `AT.01` and `WM.01` both filled
capabilities that had **zero specs two days ago**, `LG.15` turned the project's
sharpest constitutional tension into a falsifiable claim, and the `DP.04` probe
spent 530 s to prove a world edit *cannot* work rather than editing the world
hopefully. `WM.01` is an honest FAIL on a lean-PASS forecast and `LG.15` an
honest VOID on a lean-FAIL forecast — two wrong forecasts in one night, which is
what pre-registration is for.

The converse is where the drift lives and it is unchanged. `coverage`:
**0 commitments with no declared spec**, **3 CLAIM-DEAD** (smell,
shelter/building, thermal — every claim spec parked or foreclosed, and `D42`
fired *hold* on naming a successor), **14 with live claim specs and nothing
passing**. The three the charter names as most likely neglected:
**curiosity 2 of 12**, **one-brain/unison 2 of 29**, **memory-across-lives 2 of
10**. `touch`, `tool use`, `told world`, `proprioception`, `sleep` and
`hunger/thirst` all read **0 passing**. `goal_unrunnable = 7`.

**4. Is the builder alive and productive — YES, and this is the strongest 7-hour
stretch on record this month.** Seven consecutive `rc=0` (18:07 → 00:15), zero
lost slots, zero pace skips, no repeated identical failure, gate `week:all
models` **38%** at the last slot (the line acted on; `week:Fable` 69% is not the
gate), pace allowance ~66.6% at 64% week-elapsed — wide open. The builder
inherited a dirty tree three times and correctly identified all three as the
regate lane mid-sweep rather than work to adopt. **Creature gate: NONE for the
22nd consecutive slot** — re-derived by the builder each time, with `T2.01`,
`XL.01` and `T6.01` all behind desk-owned rows. That streak is the honest
headline: the loop is working perfectly on everything except the three gates
that would put Jack in his world.

**5. Compute honesty — the GPU side is a sixth consecutive unspent week and the
cause is a pipeline, not a gate.**

```
2026-W41   0.000 h of 30.0 drawn    30.0 h expired unbought
2026-W42   2.078 h of 30.0 drawn    27.9 h expired unbought (0.083 ok + 1.995 failed)
2026-W43   1.777 h drawn, 1 job, all ok   ~28.2 h remain, expiring Saturday 2026-10-31
```

W43's single job bought `T2.05` attempt 5 — a GPU hour with a ledger row and a
pre-registered reading, which is the standard. `gpu_hours_no_verdict` **53.24 h**,
of which `D1.0` is **33.78 h / 2 attempts / 0 verdicts** = 63% of all unredeemed
GPU time on one spec, unchanged. The reason nothing is dispatched is not the
gate: both GPU specs that could absorb hours (`T2.03`, `T2.14`) read
`needs T1.08`, which is FAIL with a sixth attempt prohibited in writing — and
note from RANK 2 that those are two of the seven dead-dependency certificates.
**The builder correctly manufactured nothing**, stating in three separate slot
summaries that W43 has *"no legal buyer."* That is the right conduct and the
wrong outcome.

CPU side: today's meter is clean so far. The 161st audit's RANK 1 (a legal
dispatch billing the day meter at 1/3 of its true core cost) is discharged on the
comment half at `434b7ed`; the constant question stays `D39`'s.

**6. Stuck decisions — the `D1` disease is absent; nothing was armed because
nothing is unarmed.** `MEANS-ESCALATED` **zero** — no fork a measurement could
settle sits on the owner's desk. `UNDECLARED` **zero**, so the charter's *"arm at
least one per audit"* had nothing to act on and the ratchet reads `0/0`;
reported rather than silently skipped. `OVERDUE — DEFAULT IS DUE TO FIRE`
**zero**. Three armed goal-class entries are live and not yet due (`D51`
2026-11-03, `D53`/`D54` 2026-11-08), each with a monotone default that moves no
threshold. `firing transcriptions` **35 of 35**. Nothing was acted on without
being recorded.

The standing defect is the 161st audit's RANK 3 and it has aged a day: **eleven
`CONDUCT-DESK` entries, seven of them past their `decide_by`** — `D33` **33 d**,
`D35` 32 d, `D38` 22 d, `D40` 16 d, `D39` 11 d, `D46` 9 d, `D47` 2 d — all armed,
none able to fire, because `decisions.py:1469-1477` keeps `conduct` entries out
of the only lane that emits the OVERDUE class. `decisions_default_action_expired`
reads **0, at its floor**. Routed yesterday as
`conduct-desk-expiries-reach-no-reader-no-clock-and-no-exit-code` (`DUE
2026-11-05`); **not re-routed**, and I did not fire any of the seven for the
reason the 161st gave at source and I re-derived: three are the Review's own
sitting order and `D39`'s own text says no desk may answer it.

**7. Bakeoff hygiene — two standing violations, both recorded, both at floor.**
`champions --check` EXIT 0, ratchet ok: `VERDICT-IS-A-VOID` on **Learning core**
(held BY VERDICT off `LC.03`, whose ledger row is a VOID — the "VOID treated as
a verdict" this section asks about, carried on its face), and **World**
`VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED`, held by the file's strongest marking
while naming no deciding run and no re-open trigger. `champions_trigger_debt = 3`
and `champions_unwinnable = 4`, both unchanged and at floor. **No winner was
chosen inside a noise margin this week.** `ARENA-MISSING` is **0** — and note the
overseer prompt still says *"8 seats today"*, which is a dated example, not
current state. One decision made by an arithmetic rather than a measurement:
`D42`, `D43`, `D44` all fired their *hold* branches, which is the conservative
direction in each case.

The 161st audit's RANK 4 (`LC.08` absent from the Learning-core seat's `VERDICT:`
and `TRIGGER:` fields) is **still open** — `champions --check` prints `LC.00 …
LC.07` and `LC.08` appears nowhere in the 68-member arena list. Owned by
`learning-core-arena-stops-at-lc07-and-lc08-has-run` (`DUE 2026-11-03`), whose
discharge the row says belongs in the same act as the `lc08` stamp. The stamp has
not happened: `lc08-comparator-registered-and-unbought` still reads **OPEN,
DELIVERED — AWAITING STAMP at `b34de2e`, DUE 2026-10-26** — today.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?** **Yes, and by more than yesterday — but the gain is in the body of the
machine and the gate that matters has not moved in 22 slots.**

What is genuinely better: the `fast/slow` commitment went from never-measured to
measured (`WM.01`, an honest FAIL on a lean-PASS forecast); the `attention`
commitment went from zero specs to a clean PASS whose must-fail controls both
held per seed (`AT.01`); the project's sharpest constitutional tension — is the
character sheet a costume? — went from prose to a registered experiment that
**ran** and returned a finding sharper than the forecast (`LG.15`: the shipped
answering path answers nobody at either end, and that is a creature-shaped fact
about whether he can be honestly tested at all). Two wrong forecasts in one
night, both disclosed. Seven `rc=0` slots, no lost work, no manufactured
dispatch. That is a good night by any standard this project has.

Against it, and this is most of the page: of 261 specs, 108 are demonstrated and
**96 are unreachable**. **Fourteen** of the owner's commitments have live claim
specs and **nothing passing at all** — `touch`, `tool use`, `told world`,
`proprioception`, `sleep` and `hunger/thirst` among them — and **three more are
CLAIM-DEAD** with every successor PILOT-BLOCKED and `D42` fired *hold* on naming
one. The unison claim reads 2 of 29, curiosity 2 of 12. Thirteen of the 108 certificates stand on foundations that are
either unresolvable by their own record (RANK 1) or permanently exempt from
re-measurement (RANK 2), and `pass_on_dead_dependency` sits at 7 against a floor
of 3 with **no reachable path down**. Twenty-eight expiring GPU-hours have no
legal buyer for the sixth week running. And the creature gate — `T2.01`, `XL.01`,
`T6.01`, the three specs that would actually put him in his world — has read
**NONE for 22 consecutive slots**, every one correctly derived, every one blocked
behind a desk whose queue grows by 3.71 rows a cycle and disposes of 2.86.

**The sentence I would keep if I could keep one.** Last night this project
measured, for the first time, that its own shipped question-answering path
produces nothing at either end — and it found that out because a spec was written
to ask whether a character sheet was a costume. That is the ladder working:
asking a question about Jack and getting an uncomfortable answer about Jack.
Then this audit found that six standing certificates cannot rule out a failing
seed, seven cannot be re-measured at all, and the tool that knows the first
number has not been run by anything in twenty-five days. **Both halves are true
in the same night, and the honest reading is that the measuring is now good
enough to show us how much of the ladder is resting on things we stopped
checking.**

---

## THE BILL

Editing `docs/REVIEW_QUEUE.md` stales `T0.21`, `T0.28` and `T0.31` (the three
specs that declare it). Priced before the edit, not discovered after: all three
are seconds-class re-buys (attempt 22 each, last bought `906ee31` at 00:15).
**I held the edit until the regate sweep ended** so the 160th audit's
dirty-mid-sweep class could not reproduce, then re-ran every instrument and
committed by filename, never `-A`. The re-buy is the builder's next-slot
housekeeping, as it has been for the last four queue edits.

## FOR THE BUILDER

Ordered cheapest and most perishable first. Every number re-derived this audit.

1. **Nothing of mine is owed before your next slot.** All three of my findings
   are `ROUTED:` rows, deliberately: each one's discharge condition outlives this
   page, which is rewritten at 06:37. Read `docs/REVIEW_QUEUE.md`, not this
   section, for what has a clock.
2. **Pay the queue bill: `T0.21`/`T0.28`/`T0.31` re-buy from a clean tree.**
   Three seconds-class runs. Check the tree is clean first — a regate sweep was
   running at 00:43 and the last two slots both stamped a `+dirty` re-buy by
   starting one before their own commit landed.
3. **`run worst-seed-audit` is a one-command read that nothing in this project
   invokes, and it currently names six standing PASS certificates.** Not an
   order and not a repair — RANK 1's row owns the repair, and narrowing the
   audit's domain to lower a count is refused on it in advance. Named because
   you can read it in two seconds and no organ has in 25 days.
4. **`lc08-comparator-registered-and-unbought` is DUE today and reads OPEN /
   DELIVERED — AWAITING STAMP at `b34de2e` with no `PRE-DATED RECEIPT` flag.**
   The stamp is the desk's, not yours — do **not** redo the buy. Named because
   the champions-entry repair (`learning-core-arena-stops-at-lc07-and-lc08-has-run`)
   is supposed to ride in the same act, and both are perishable.
5. **`AT.01`'s per-seed-booleans-from-authorship idiom is the thing to keep
   doing.** It is why `WM.01`'s defect cannot recur on the newest spec, and it
   is the cheapest version of RANK 1's repair available to an author.

## FOR THE OWNER

**1. NO-DECISION — seven of your PASS certificates cannot be re-measured, and
the rule that makes them safe is the rule that makes them permanent.** Nothing
here is yours to rule; it is routed with a date. One paragraph deserves your eye
because no page has shown it before. `pass_on_dead_dependency` reads **7 against
a declared floor of 3** — `T0.19`, `T2.03`, `T0.18`, `T2.14`, `LF.02`, `T1.12`,
`PL.00`. The regate rotation, the only organ that routinely re-tests a standing
certificate, has skipped four of them for **96 consecutive sweeps** because
rotating them would record `BLOCKED` rather than a verdict — which is the
*correct* refusal, and is also why `T0.19`'s PASS has not actually run since
**2026-08-11, 76 days ago**. Three of the seven hold champion seats, including
the **PLASTIC-ONLY decree** seat your 2026-08-09 ruling created. And the floor is
unreachable by construction: all five blocking dependencies are red, four with no
reachable repair and one (`T1.08`) with a sixth attempt prohibited in writing.
I may strengthen a bar and I may never weaken one, so **I am not proposing the
floor move** — I am telling you that `7 vs 3` is not a backlog that drains.

**2. NO-DECISION — the instrument built to catch wrong verdicts has never been
run by anything, and a wrong verdict got through last night.** `WM.01` recorded
PASS on a mean of 0.1927 over per-seed margins {0.4707, 0.0800, 0.0273} against a
bar its own docstring had pre-registered as *"on every seed."* **Your builder
caught it in eleven minutes, shipped a strictly harder gate, re-ran, and recorded
an honest FAIL — I want that said first, because it is the system working.** What
I am reporting is that `run worst-seed-audit`, built 2026-10-01 for exactly this
class, is in no cron, no script, no prompt order, not in `run status`, has no
ratchet counter, always exits 0, and last reached a committed document 25 days
ago. Live it names **six standing PASS certificates whose record cannot rule out
a failing seed** (`LT.01`, `ME.10`, `NE.00`, `PG.4`, `PS.02`, `T2.20`) — of which
exactly one, `PG.4`, carries a caveat a board reader can see. The design ruling
that would make the class unrepeatable (`aggregate-hides-worst-seed` STEP 2) is
your desk's, falls **due today**, was dated onto 10-26 because *"10-26 carries
nothing"*, and 10-26 now carries seven rows and an `AMBER: pile`.

**3. NO-DECISION — a tenant-safety breach, reported because `D18` fired
*MEASURE AND REPORT* and the report reached one log line.** At 21:23 last night a
detached child peaked at **1736 MB against the 1536 MB ceiling** — 113%, on the
box your four customer agents share. Nothing was killed and nothing should have
been: `D18`'s armed default is *GATE NOTHING, RELAX NOTHING*, and every constant
is byte-unmoved. The finding is that the breach is in no ledger row (the process
never called `run_spec`, so it wrote neither a row nor a receipt), in no
committed document (the loop emits the line *after* the slot summary is written,
so the builder could not have carried it), and in no counter — `D18`'s own firing
record states the population as *"8 of 69"* and live it is **10 of 164**, a
number nothing in this repo can print. Routed with a date, report-half only.

**4. NO-DECISION — the perishable compute cost, per `D30`'s standing default.**
`2026-W43` stands at **1.777 h drawn of Kaggle's 30 h free quota — ~28.2 h free,
expiring Saturday 2026-10-31**, the **sixth consecutive week** it expires
substantially unbought. The cause is not a gate and not a blackout: the builder
ran seven consecutive `rc=0` slots last night with the weekly meter at 38%, and
both GPU specs that could absorb hours (`T2.03`, `T2.14`) sit behind `T1.08`.
It is a pipeline cost, which is a different repair. The builder correctly
manufactured no dispatch and said so three times.

**5. NO-DECISION — one thing went right and this page is structurally biased
against saying so.** Three of your constitutional commitments got their
first-ever measurement in a single night. `fast/slow` — *"it must still be
connected but slightly different purposes… it must be in the research and
tests"* — had never been measured in this project's history; it now has an
honest FAIL that says the working-memory buffer learns composition on one seed of
three. `attention` had **zero specs two days ago**; it now has a PASS where the
modality gate cut vision's weight from 0.459 to 0.214 when vision went dark, with
both must-fail controls holding on every seed. And `LG.15` asked whether the
character sheet is a costume and discovered something worse and more useful: the
shipped answering path answers nobody at either end. **Three forecasts were made
and two were wrong**, which is the only evidence that any of them could have
failed.
