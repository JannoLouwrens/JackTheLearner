# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-25 18:37–19:0x UTC — 161st audit.** VERDICT: **DRIFTING.**

**Three disclosures first, because each changes how a number below should be
read.**

1. **A regate sweep opened at 18:43 and is live in this tree as I write.** It
   re-bought `T0.21`/`T0.28`/`T0.31` at 18:43–18:45 — all three before my first
   edit — and is now in the standing-PASS rotation. My edits touch
   `docs/REVIEW_QUEUE.md`, which only those three declare, and doc dirt is
   per-spec, so the 160th audit's RANK 2 class (a tree going dirty mid-sweep)
   does not reproduce on this sitting. I re-ran every instrument after the edits
   and committed **by filename**, never `-A`.
2. **The two pages I read as inputs are both seal-sealed drafts.**
   `docs/OVERSIGHT.md` was sealed at 12:57 (`fca2035`, overseer rc=1) and
   `docs/PROGRESS.md` at 13:02 (`ca62a3a`, Review rc=124). Both pages are in
   fact *complete* — `PROGRESS.md` carries its closing receipt line, both
   sections and both asks — and the builder acted on orders from both at 13:24
   and 14:16. So the banners are honest about the exit code and inert as a
   steering control. Named, not routed: the seal is correct and a page that is
   finished is finished.
3. **The 160th audit's five `FOR THE BUILDER` items are all discharged**, which
   is the best thing on this page. Verified at HEAD rather than from commit
   messages: `D44`'s firing transcribed (`be670f0`; `decisions --check` now
   reads *35 of 35*), the heading reader widened to containment with a replay
   fixture (`c92f5ba`; `run status` reads **17 items on 3 of 3 declared pages**,
   this organ's page no longer invisible), `LC.08`'s amendment blind spot
   declared as a priced limit (`4032e9c`), and `LC.08` itself bought and
   harvested (`b34de2e`) — **106 → 107 demonstrated**, the first first-ever
   verdict in five weeks that is about Jack and not about the apparatus.

True exit codes, taken from the processes and not through a pipe, after my own
edits: `coverage` **2** · `decisions --check` **0** · `champions --check` **0** ·
`run review-queue` **0** · `run status` **2** · `run ratchets` **2**.

Ratchet slot line, re-derived after my edits:

> `8 MOVED (fail_unowned_owned_forms {'disposed': 1, 'queue-row': 27,
> 'repaired_by': 1} -> {'disposed': 2, 'queue-row': 27, 'repaired_by': 1},
> gpu_hours_no_verdict [map], pass_on_dead_dependency 6 -> 7,
> projection_vs_estimate [+1], review_queue_net_arrivals -5 -> -2,
> review_queue_piled_on 15 -> 5, review_queue_violation_forms {'OVERDUE': 16}
> -> {}, review_queue_violations 16 -> 0); 1 day-rolled (cpu_foreclosed_now);
> no counter refused to compute; floors: 2 ABOVE (pass_on_dead_dependency,
> unreachable), 0 BELOW, 0 UNVERIFIED.`

`claim_dead = 3` · `commitments_uncovered = 0` · `fail_unowned = 6` (at floor) ·
`champions_trigger_debt = 3` · `champions_unwinnable = 4` (at floor) ·
`goal_unrunnable = 7` · `dark_slots = 0` — and the last one is wrong again
today, for a mechanism nobody had found: see RANK 2.

---

## RANK 1 — a legal dispatch billed the tenants' box at one third of what it took, and the ceiling's own safety premise says that cannot happen

**This is the most serious class this organ reports, because SYSTEM.md and
`CLAUDE.md` both rank tenant safety above the ladder, and because it is a
measured fact from today rather than an arithmetic worry.**

`experiments/cpu_budget.py:81-88` justifies `CPU_DAY_CEILING_S = 57600.0` in its
own words:

> *16 h: ... (b) **children are single-process Python**, so 16 h is <= 1/6 of
> this 4-core box's core-day*

**That premise was falsified today by a dispatch every gate admitted.**
`LC.08`'s detached buy ran **three concurrent workers** at ~1 core each — the
launching organ's own receipt, written before any result. Read off the ledger
row and the meter, not argued:

| reading | value | source |
|---|---|---|
| `core_s` per seed | 17280.38 (std 0.04) | `LC.08` metrics |
| seeds | 3, concurrent | row + launch receipt |
| core-seconds actually spent | **51841** | 17280.38 × 3 |
| billed to the day meter | **16912.43 s** | `cpu_budget.json`, 2026-10-25 |
| undercount | **3.07×** | 51841 / 16912 |

**And the ceiling's own stated budget was exceeded while its meter read 43%.**
`57600 s` is equated with 16 core-hours via the single-process premise (1/6 of a
4-core core-day = 16 core-h). Today:

```
charged today        24741.82 s of 57600   =  43% of the ceiling
  of which LC.08     16912.43 s wall       =  14.40 core-h actually spent
  the rest            7829.39 s wall      >=   2.17 core-h  (single-process FLOOR;
                                                this file itself warns 1.68-1.72x
                                                for a 2-thread child)
real core-hours given up                   >= 16.58 core-h  vs the comment's own 16
```

The guard that **did** hold is the instantaneous one: `admit_detached` refuses
above `LOAD_CEILING` 6.0, and the loop's own iteration-start readings across the
run went 0.10 → 3.23 → 3.49 → 3.97 → **4.87** (`ladder.log`, 13:07→17:07). 4.87
on 4 cores is 122% of this box's cores, held for hours. Nothing was breached;
the point is that the ceiling's premise was written so it would not have to lean
on the load gate, and today it did.

**The governing decision already exists and its clock is dead.** `D39` is this
exact question — *what should the CPU day-ceiling count?* — and its arithmetic is
quoted throughout for *"a single-process child occupying **one core of four**."*
Today's legal child occupied three, so **the entry understates its own subject by
3×**. Its `decide_by` was **2026-10-15**; it has passed, its armed default
(status quo) has not fired, and it cannot — see RANK 3.

**A second reading, offered as LATENT rather than inflated into a breach.**
`child_estimate_s` now returns, for this project's two largest CPU spends ever:

```
LC.08   measured 0.03 s  ->  (10.12, "MEASURED 0.03s x4 + 10s")   true cost ~14.4 core-h
LC.03   measured 0.02 s  ->  (10.08, "MEASURED 0.02s x4 + 10s")   true cost ~190 core-h
```

Both read `duration_s`, which the detached harvest writes as the wall clock of
the *harvest call*. So the admission projection is ~5100× too low for `LC.08`
under a provenance string asserting `MEASURED`. **It is biting nothing today**
and the two reasons are stated so the next reader need not re-derive them: the
runner lane categorically refuses `cpu<48h`, and `admit_detached` is *"Est-free
by design."* **What deserves attention is the direction.** The 69th audit proved
this projection one-sided safe — *"may only TIGHTEN an estimate ... can never
refuse something it would have admitted before"* — and `T0.33` property 12
asserts that over the registry. Every word is about **false refusals**. The
companion scar in the same file is also a refusal (`SO.08` foreclosed a day at
~28,000× its measured cost). Both the proof and the scar examine the direction
that costs a dispatch; the direction that costs the tenants' box has no proof,
no scar and no reader.

**ACTED, not just reported.** Filed as an **EVIDENCE ADDENDUM on `D39`** — a
premise correction in the established form, moving no threshold, proposing no
fifth option, and expressly declining to recommend option (ii), because
core-seconds against 4 cores would RELAX a tenant protection and this organ may
not propose a loosening. `CPU_DAY_CEILING_S` 57600, `LOAD_CEILING` 6.0,
`rtf.BUDGET_SECONDS["cpu<48h"]` 172800, `PROJECTION_SAFETY` 4.0 and
`CHILD_OVERHEAD_S` 10.0 are byte-unmoved. `LC.08`'s certificate stands as
bought: this is a finding about the meter, not about the run or the organ that
launched it.

## RANK 2 — both of the two things that record a lost builder slot erase themselves on recovery, and one of them is a single line

`dark_slots` has read **0** through two consecutive audits' worth of outage, and
until today the explanation on the record was that it is a trailing streak. That
is half of it. The other half is that **the file the loop tells readers to
consult is deliberately emptied.**

`scripts/ladder_loop.sh:406-408`:

```sh
if [ "$RC" = 0 ] && [ -s "$LOST" ] && ! limit_hit; then
  say "recovered — clearing $(wc -l < "$LOST") lost-iteration marker(s)"
  : > "$LOST"
fi
```

Read live: `/data/jack-logs/lost_iterations.log` is **0 bytes**, mtime
**2026-10-25 18:12** — the exact second of the recovering slot's `rc=0`.

**Today's worked instance, counted by hand off `ladder.log`.** Three consecutive
slots were lost — **15:07, 16:07, 17:07**, each `rc=1`, each reading
*"You've hit your session limit · resets 5:50pm (UTC)"* on fable, then opus,
then sonnet, then *"SESSION LIMIT on every model — marking the lost
iteration."* The 18:07 slot inherited three markers, bought `LC.08`, and closed
with *"recovered — clearing 3 lost-iteration marker(s)."* **After that line
nothing tracked by git, nothing in `experiments/`, and no counter anywhere holds
the fact that three slots were lost.** This is a different mechanism from the
weekly 90% stop the last two audits reported: the weekly meter read 27–33%
throughout, and the cap that fired was the 5-hour session one, which no gate in
this project samples.

The cause is a good intention, which is why it survived. `:370-371` states the
design — *"The marker is the point — a dead iteration must be a number, not a
silence"* — and the clearing exists so a stale marker cannot make a healthy loop
report an outage. The comment at `:379-382` even records that the **write** half
failed once (*"`lost_iterations.log` read 0 bytes through the whole 08-21
outage"*, Review 2026-08-27). The write was repaired. **The erase was never
examined.** The 160th audit measured the consequence — *"Nothing in
`experiments/` holds any trace of the 72 slots"* — and did not find the cause.

**Routed** as `limit-lost-slots-are-erased-by-both-of-their-own-recorders`
(`DUE: 2026-11-03`), explicitly NOT a re-route of the open
`dark-slot-ratchet-is-a-streak-...` row — checked at source: that row's subject
is the ratchet's shape, this one's is the artifact, and repairing the ratchet
alone leaves it nothing to read. The repair is constrained on the row to
SEPARATING the two uses of one file, because the live-marker semantics must
survive; deleting the clearing outright would make every recovered slot report
an outage.

## RANK 3 — seven armed defaults are past their dates, oldest 32 days, and no instrument in this project can fire one of them

| entry | `decide_by` | days past | armed? |
|---|---|---|---|
| `D33` | 2026-09-23 | **32** | yes (1853 ch) |
| `D35` | 2026-09-24 | 31 | yes (513 ch) |
| `D38` | 2026-10-04 | 21 | yes (912 ch) |
| `D40` | 2026-10-10 | 15 | yes (780 ch) |
| `D39` | 2026-10-15 | 10 | yes (681 ch) |
| `D46` | 2026-10-17 | 8 | yes (1050 ch) |
| `D47` | 2026-10-24 | 1 | yes (855 ch) |

Every one is armed, so `NO-DEFAULT` is correctly silent. **Why nothing fires is
in source and it is deliberate.** `decisions.py:1469-1477`: *"`rows` is the
owner's armed-decisions report and owner-owed tracking, which a conduct entry
must never join (its visibility lane is CONDUCT-DESK)"*, and the branch
`continue`s before `rows` is built. `rows` is the **only** lane that emits
`OVERDUE — DEFAULT IS DUE TO FIRE`, whose own instruction is *"Fire the default,
journal it loudly ... a deadline that moves when it is reached is the deadlock it
replaced."* So for a `conduct` entry the deadline is reached and nothing moves.
The design is defensible on its own terms; its consequence was never priced.

**And no counter sees it.** `decisions_default_action_expired` reads **0, AT its
floor** — it measures a different thing (a default whose *text* names a date
already past at firing time) — and `decisions --check` exits **0**.

**The reporting history is the second half of this finding, and it indicts my own
two predecessors.** The 159th reported six, the 160th reported seven as `RANK 6`
and in `FOR THE OWNER`. Both reported it as **prose on this page**, which is
rewritten at 00:37, 06:37, 12:37 and 18:37 — so a 32-day breach was written down
and overwritten at least six times in two days. That is precisely what the
2026-10-01 ruling on `oversight-for-the-builder-has-no-reader` forbids, and the
rule is on this organ's own charter. **Routed** as
`conduct-desk-expiries-reach-no-reader-no-clock-and-no-exit-code`
(`DUE: 2026-11-05`).

**I did not fire any of the seven, and the reason is specific rather than
reflexive.** Three concern the Review's own capability and sitting order
(`D33`, `D38`, `D47`) and no other organ can execute them. `D39` is the sharpest
member and the one that proves the class is structural rather than clerical: its
own text says the `CONDUCT-DESK` label *"is wrong about this entry and right to
keep printing"*, because SYSTEM.md law 4 puts a tenant-protection threshold
beyond every desk **and every default**. So `D39` is a `conduct`-classed entry
that is genuinely the owner's, and `CLASSES = ("means", "conduct", "goal")` has
no value that can say *"no desk may answer this."* Firing it by arithmetic is the
one move its own author forbade in writing.

## RANK 4 — the Learning-core seat cannot see the spec built to contest it, and the window to connect them closes tomorrow

`docs/CHAMPIONS.md:377` declares, field by field: `HELD: BY VERDICT`,
`VERDICT: LC.03`, `TRIGGER: LC.07, LC.03, UB.10`, `ARENA: LC.00–LC.07`.

`LC.08` was registered on 2026-10-20 by a ruling that calls it, at
`registry_expansion.py:2456-2460`, *"The Learning-core seat's only live
measurement ... the one instrument that will ever say whether the objective A4 is
NAMED for produced the margins it was seated on."* **It is in neither field.**
`champions --check`'s own arena expansion for this seat prints `LC.00 … LC.07`,
and `LC.08` appears nowhere in its 68-member `arenas that DO exist` list.

So in one run, on one seat, today: the tool reports `VERDICT-IS-A-VOID` and
`TRIGGER-UNREACHABLE` — *"every pre-registered re-open trigger is a closed
door ... the promise of contestability has decayed to zero"* — and exits **0**
with `ratchet ok`. Both statements are correct about the three ids the file
declares. Neither can see the fourth, which is the only one that produced a live
reading on this seat in eight weeks.

**This is a defect and not bookkeeping, because `LC.08` could have moved the
seat.** Its `falsified_by` branch is explicit: both margins reproducing without
`l_bind` would have meant *"149,312 params — 17.3% of the seated arm — plus the
objective it is named for are not what the margins measure."* It came back the
other way, on both of the two statistics the seat was scored on: a deficit of
**116.14** against each, versus the seated arm's own seed spreads of
`lg_margin_null_std` 37.33 and `lg_margin_twin_std` 42.25 — the pre-registered
PASS branch on both conjuncts, with `deadweight_reading` 0.0. **The seat is
vindicated and nothing in the seat file or the seat tool records that it was
tested.** Today the error direction is the
conservative one — the seat reads deader than it is. Had the result gone the
other way, it would have been an arena result no seat tool could credit.

**Routed** as `learning-core-arena-stops-at-lc07-and-lc08-has-run`
(`DUE: 2026-11-03`, and a deadline rather than a schedule: it is dischargeable
tomorrow in the same act as the `lc08` stamp, which is where it belongs). The row
**refuses in advance** any version of this edit that lowers a champions count:
the seat is still held off a VOID and `LC.07` is still `PILOT-BLOCKED`, so if the
edit would discharge `TRIGGER-UNREACHABLE`, `VERDICT-IS-A-VOID` or
`UNVERIFIED VERDICTS`, the edit is wrong and not the findings.

## RANK 5 — the week's one certificate about Jack cannot say what code computed it, in the lane whose sibling got that exact repair five days ago

`LC.08` records `commit: e7d2475` — HEAD at **17:08** — for a run launched at
**13:19** and written at **18:00:57**. Four commits landed inside the window
(`c92f5ba`, `704bc71`, `b7f87a5`, `e7d2475`). The detached CPU lane records the
HEAD at *harvest* time, not the HEAD the work was computed from.

This is the same finding as `gpu-receipt-head-is-push-time-not-kernel-time`,
which was routed 2026-09-30 and `ACTED` 2026-10-20 at `c08d15a` with both halves
— **(i) PREVENTS**, the dispatch HEAD resolved to a full sha and pinned as `ref`
(`gpu.py:364`), and **(ii) DETECTS**, the kernel printing `CLONED_HEAD` and the
harvester refusing on mismatch (`:1258-1272`). Read at source this sitting:
**every line of that repair is in `experiments/gpu.py`.** The detached CPU lane
has neither half.

**What IS protected, stated so this is not read as bigger than it is:**
`impl_sha` (`c9ede74c928bbb27`) covers the test file plus every declared
`IMPL_DEPS` path, and `dirty_files` is `null`. The implementation that ran is
pinned and the tree was clean. What is not pinned is everything else the
`commit` field is read as naming — a reader doing `git checkout e7d2475` to
reproduce `LC.08` gets a tree four commits ahead of the one that computed it.

**Why it is perishable.** The launch head exists in exactly one place:
hand-typed prose in a `BUILDER-TRACE:` on `lc08-comparator-registered-and-
unbought`, which renders under `DELIVERED — AWAITING STAMP` with
`DUE: 2026-10-26`. `ACTED` is terminal. **Routed** as
`detached-cpu-receipt-records-harvest-time-head-not-launch-time-head`
(`DUE: 2026-11-05`), with re-stamping `LC.08`'s row refused in advance.

---

## Section-by-section, including the clean results

**1. Integrity of the ledger — CLEAN, and it is the strongest result on this
page.** All **107** PASS rows checked mechanically, including the one bought 40
minutes ago: every declared implementation file exists, **every recorded `commit`
resolves in git (0 dangling)**, and every spec declares a `control`. The two that
read controlless are `T0.01` and `T0.10`, and both carry explicit
`NoControlByDecision` objects quoting the 52nd audit's reasoning — verified by
type, not by name. The ledger holds 161 rows against 260 registered specs.

**One caveat on the newest certificate, and credit where it is owed.** `LC.08` is
**the first and only row in this entire ledger** whose `message` records
`HASH-SALT DIFFERENTIAL SKIPPED (18 deciding metric(s)): experiment fn is not an
importable module-level function; salt dependence unmeasured for this row`. The
reason is structural, not transient — a re-run buys the same skip. **The builder
disclosed it unprompted in both durable places**, on the queue row (*"per the
W1.01 precedent do not quote the PASS as salt-robust"*) and in its slot journal,
which is exactly right and is the opposite of the failure this organ exists to
catch. What remains is that no instrument can see it: the open row
`salt-differential-has-no-coverage-counter-and-an-unsplit-d27-exemption`
(`DUE: 2026-11-02`) says so in terms, and once `lc08-...` is stamped `ACTED` the
only durable trace is a ledger `message` string nothing reads. Not re-routed —
that row owns it — but it is the single most important thing for the Review to
carry forward into the champions entry tomorrow.

**2. Thresholds and controls over time — NO LOOSENING FOUND.** The 160th audit
covered the 7-day window to 12:37 (3,015 lines over 16 commits, no loosening).
I re-read the delta since its seal independently; two commits touch
registry/test code and both move in the strengthening direction:

- `c92f5ba` widens `steering.py`'s `_BUILDER_HEADING` from an exact-line match to
  containment. **Strictly widening a reader's domain** — it can only find more
  orders, never fewer — with the `\b` guard retained and a new selftest fixture
  that replays the real 10-22 outage (0 → 7 items) while requiring
  `FOR THE BUILDERS` and prose mentions to stay silent. The known cost is
  declared in the diff rather than discovered later. No bar moved.
- `4032e9c` adds a **DECLARED LIMIT** comment to `LC.08`'s `SEAT_RECORDED` pins:
  the fail-safe covers a forbidden re-run and not a ledger amendment, accepted
  with the acceptance priced from `Ledger.amend`'s own lanes. A declaration, not
  a threshold.

Live spot-check rather than inherited: `RANDOM_DWELL_MAX` reads **0.0185** at
HEAD, tighter than the 0.02 the `T3.06` VOID was read against.

**3. Drift from the goal — the day's work is on-goal, and the converse has not
moved.** Nine builder slots since 10:07; six `rc=0`, three lost to the session
cap (RANK 2). PASS delta **+1** (106 → 107), and it is the right +1:

| slot | unit | which `GOAL.md` sentence |
|---|---|---|
| 13:24 | `LC.08` buy launched (14.4 core-h detached) | *"components that must EARN their parameters via ablation or be deleted"* |
| 14:16 | heading reader widened; dead `AWAITING` row cleared | the apparatus, not Jack |
| 18:12 | `LC.08` harvested **PASS** | *"every sense is load-bearing (and we PROVE each one is)"* — the same discipline applied to an objective |

`LC.08` is the day's real movement and it is a genuine ablation result: with
`l_bind`'s gradient zeroed on an init-identical architecture at the identical
paired venue, `life_gain` collapsed +96.83 → −19.31, a 116.14 deficit against
seat seed-stds of 37.33 and 42.25. **The spec's own pre-registered forecast
leaned FAIL and the measurement went the other way** — which is what a test that
could have failed looks like.

The converse is where the drift lives and it is unchanged. `coverage`:
**0 commitments with no declared spec**, **3 CLAIM-DEAD** (smell,
shelter/building, thermal — every claim spec parked or foreclosed), **14 with
live claim specs and nothing passing**. The three the charter names as most
likely to be neglected read **curiosity 2 of 12**, **one-brain/unison 1 of 29**,
**memory-across-lives 2 of 10**. `NO-LIVE-PATH` stands at 6 distinct holes, every
repair a REGISTRATION. And `coverage` reports **4 NEW unrunnable `GOAL.md`
citations** — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all `welded<-LC.07` — each
id resolving to a corpse while the citation's present tense says otherwise;
owned by the open row `gen-four-revival-needs-an-affordable-lc07-successor`.

**4. Is the builder alive and productive — YES, and better than yesterday.** Six
`rc=0` in nine hours, three slots lost to a 5-hour session cap that self-cleared
at 17:50, PASS delta +1, every one of the 160th audit's orders discharged, and no
identical repeated failure. The weekly gate is wide open (`week:all models`
**34%**). The one honest criticism is the one RANK 2 names: the loop's own record
of the three losses was deleted by the loop.

**5. Compute honesty — the CPU side is RANK 1; the GPU side is a sixth
consecutive week of unspent free quota, and the cause is not the loop.**

```
2026-W41   0.000 h of 30.0 drawn       30.0 h expired unbought
2026-W42   2.078 h of 30.0 drawn       27.9 h expired unbought  (0.083 ok + 1.004 + 0.991 failed)
2026-W43   1.777 h drawn so far       ~28.2 h remain, expiring Saturday 2026-10-31
```

The single W43 job bought the `T2.05` attempt-5 row — a GPU hour with a ledger
entry and a pre-registered reading, which is the standard. `gpu_hours_no_verdict`
reads **53.24 h total**, of which `D1.0` is **33.78 h across 2 attempts and 0
verdicts** — 63% of all unredeemed GPU time on one spec, unchanged.
`projection_vs_estimate` gained one row this week (`T2.05`: authorised 1.0 h,
declared `est_hours` 1.2 h) — an over-estimate against its authorisation,
declared rather than hidden. The reason nothing is being dispatched is a pipeline
fact and not a gate: both GPU specs that could absorb hours (`T2.03`, `T2.14`)
read `needs T1.08`, which is FAIL with a sixth attempt prohibited in writing.

One measurement worth recording without an accusation attached: the regate lane
refused **nine** specs this sweep for declaring a cost class their own
measurements refute — `T1.06` measured 3760 s against a `cpu<10min` label,
`T1.01` 2083 s, `T1.12` 1141 s, `XL.00` 1165 s, `T2.00` 1001 s, `PS.01` 871 s,
`LG.03` 727 s, `T1.04` 261 s against `cpu<1min`, `T0.23` 80 s. Each refusal is the
lane behaving correctly (*"stays with the loop"*), and re-labelling is an
established legal act (`1ee9c76`). Reported because the same declared classes are
what `coverage`'s queue-depth block and the day meter's worst case are computed
from.

**6. Stuck decisions — the `D1` disease is absent; the defect is that a desk's
own deadlines have no firing lane.** `MEANS-ESCALATED`: **zero** — no fork a
measurement could settle sits on the owner's desk. `UNDECLARED`: **zero**, so
there was nothing for me to arm and the ratchet reads `0/0`. `OVERDUE — DEFAULT
IS DUE TO FIRE`: **zero**, and RANK 3 is why that zero is not a clean result.
Three armed goal-class entries are live and not yet due (`D51`, `D53`, `D54`),
all with monotone defaults that move no threshold. Nothing I found was acted on without
being recorded; `firing transcriptions` reads **35 of 35**.

**7. Bakeoff hygiene — two standing violations, both recorded, both at floor,
and RANK 4 is a third that no counter reports.** `champions --check` reports
`VERDICT-IS-A-VOID` on **Learning core** — `wm-latent` held BY VERDICT, the
file's strongest marking, off `LC.03`, whose ledger row is a **VOID**. That is
exactly the "VOID treated as a verdict" this section asks about, and it is not
hidden: the seating carries its single-arm caveat on its face. The **World** seat
is worse in kind — `VERDICT-UNDECLARED` + `TRIGGER-UNDECLARED`, held BY VERDICT
while naming no deciding run and no re-open trigger at all.
`champions_trigger_debt = 3` and `champions_unwinnable = 4` are unchanged and at
floor. No winner was chosen inside a noise margin this week. The one
anchor-decided certificate (`T4.06`) still prints its own margins as fractions of
the anchor's seed spread, including the two arms that regress.

**8. The honest summary — are we closer to a curious humanoid that climbs the
ladder?** **Yes, by one real step, and for the first time in five weeks the step
is addition rather than subtraction.** `LC.08` asked whether 17.3% of the seated
learning core is dead weight, spent 14.4 core-hours finding out, forecast FAIL,
and measured the opposite: with the binding objective's gradient off, the core
stops learning to live. That is a component earning its parameters against a test
that could have deleted it — `GOAL.md`'s own standard, applied to the thing this
project will build everything else on. It is also the week's only first-ever
verdict that is about Jack and not about a tool that watches tools.

Against that, and it is most of the page: 107 of 260 specs demonstrated; three of
the owner's constitutional commitments still CLAIM-DEAD with no registered
successor; the headline unison claim at **1 of 29**; curiosity at 2 of 12; seven
standing certificates resting on recorded non-PASS foundations, four more than
the floor permits; and `unreachable` above its floor at 96. The queue's violation
class is empty and the ledger's 107 certificates all check out at source, so the
machinery is in excellent health — but the queue now holds **58 live rows at a
measured drain of 252 cycles**, and four of today's additions are mine. **The
most honest thing I can say about my own sitting is that the one certificate
bought today was about Jack, and all five of my findings were about the
instruments that watch him.** That ratio is the drift, and it is not improving.

## FOR THE BUILDER

Ordered cheapest and most perishable first. Every number re-derived this audit.

1. **Nothing of mine is owed before your next slot.** All five of my findings are
   `ROUTED:` rows or an addendum on an owner entry, deliberately: each one's
   discharge condition outlives this page, which is rewritten in six hours. Read
   `docs/REVIEW_QUEUE.md`, not this section, for what has a clock.
2. **`cpu_budget.py:81-88`'s single-process sentence is now false, and correcting
   the comment is free.** The ceiling's justification reads *"children are
   single-process Python, so 16 h is <= 1/6 of this 4-core box's core-day"*, and
   today's legal `cpu<48h` dispatch ran three concurrent workers at 3.07× the
   billed rate. **This is a COMMENT correction and nothing else** — it moves no
   constant, loosens nothing, and must not touch `CPU_DAY_CEILING_S`, which is
   `D39`'s and the owner's. Say what the ceiling actually assumes, so the next
   organ to lean on it leans on something true. Bill: price it —
   `experiments/cpu_budget.py` is named by `T0.33` and `T0.34`'s `IMPL_DEPS`
   (~2 s and ~60 s re-buys).
3. **`AT.01` and `WM.01` are still the only fresh CPU implementations on a real
   hole, and `WM.01` is the only RUNNABLE claim on the `fast/slow`
   commitment** — every other one is blocked, welded or foreclosed. `coverage`
   lists both as fillable today. Not ordered; named because after `LC.08` the
   board has no cheaper honest science on it.
4. **Do NOT re-run `LC.08` to get a salt differential.** The skip is structural
   (*"experiment fn is not an importable module-level function"*), so a re-run
   buys the identical skip plus another 14.4 core-h, and the PASS stands as
   recorded. If the salt question is to be answered it is an entry-point
   redesign, which is the open
   `salt-differential-has-no-coverage-counter-...` row's business and not a
   dispatch.
5. **`docs/PROGRESS.md` item 5 is the only steering order naming a spec the
   runner would refuse** (`DP.04`, PILOT-BLOCKED) — `run status`'s own reading,
   1 of 17 items on 3 of 3 pages. Its two-conjunct acceptance test is real work;
   executing it on conjunct (1) alone buys a predicted VOID, as that page says.

## FOR THE OWNER

**1. `D39` — I HAVE ATTACHED A PREMISE CORRECTION, NOT A RULING, AND THE NUMBER
YOU WERE GIVEN IS THREE TIMES TOO SMALL.** This entry asks what the CPU
day-ceiling should count, and it argues throughout from *"a single-process child
occupying one core of four."* Today a legal dispatch ran **three** concurrent
workers: 51,841 core-seconds spent, **16,912 billed** — a 3.07× undercount — and
the box gave up **≥16.58 core-hours** against the ceiling's own stated maximum of
16, while its meter reported 43% used. The load gate held (peak 4.87 on 4 cores,
against a ceiling of 6.0), which is the guard the premise said would not have to
carry this. **Nothing is relaxed, nothing is scheduled and no constant has moved**
— and I specifically do **not** recommend option (ii), core-seconds against 4
cores, because it would loosen a tenant protection and this organ may not propose
a loosening. The desk's recommendation of (i)+(iv) stands untouched. What changes
is that ruling (i)+(iv) now means accepting a 1/3 bill rather than a 1/1 bill.
The cheapest honest consequence needs no decision from you at all: correct the
comment, which is builder item 2.

**2. NO-DECISION — your desks' own deadlines have no firing mechanism, and the
oldest broken one is 32 days old.** Seven open `conduct` entries are past their
`decide_by` with armed defaults that cannot fire: `decisions.py` keeps
`conduct`-classed entries out of the only lane that emits `OVERDUE — DEFAULT IS
DUE TO FIRE`, by an explicit and defensible design whose consequence was never
priced. The counter for the class reads **0, at its floor**, and
`decisions --check` exits 0. **Nothing here is yours to rule** — it is routed to
the Review with a date — but one sentence deserves your eye, because it is the
reason the class is structural rather than clerical: `D39` is a
`conduct`-classed entry whose own text says the desk-executable label *"is wrong
about this entry"*, because SYSTEM.md law 4 puts a tenant-protection threshold
beyond every desk **and every default**. The taxonomy has three values and none
of them can say *"no desk may answer this."* That is why `D39` has sat ten days
with nobody able to either answer it or fire it.

**3. NO-DECISION — the gate was not what stopped the builder today, and the thing
that did leaves no trace.** Three slots were lost to a 5-hour session cap that
self-cleared, while the weekly meter the 90% stop reads sat at 27–33%. Both
mechanisms that record such a loss erase it on recovery — `dark_slots` is a
trailing streak any `rc=0` resets, and the loop truncates its own
`lost_iterations.log` the moment it recovers. Routed with a date; reported here
because the shape recurs: this project keeps discovering that the counters which
would show it losing time reset exactly when the loss ends.

**4. NO-DECISION — the perishable compute cost, reported per `D30`'s standing
default.** `2026-W43` stands at **1.777 h drawn of Kaggle's 30 h free quota —
~28.2 h free, expiring Saturday 2026-10-31**. That would be the **sixth
consecutive week** the quota expires substantially unbought. The honest cause is
no longer a blackout: the builder is awake, the weekly meter reads 34%, and the
two GPU specs that could absorb hours both sit behind `T1.08`, whose estimator
question is on a desk with a sixth attempt prohibited in writing. It is a
**pipeline** cost rather than a gate cost, which is a different repair.

**5. NO-DECISION — one thing went RIGHT today and it is worth a sentence,
because this page is structurally biased against saying so.** `LC.08` asked
whether 17.3% of the seated learning core — 149,312 parameters and the objective
they implement — is dead weight. It spent 14.4 core-hours, its own
pre-registration forecast FAIL, and it measured the opposite: with that
objective's gradient switched off, at an identical architecture, venue and seed
set, the core's `life_gain` collapsed from +96.83 to −19.31. A component was put
on trial by a test that could have deleted it, and it earned its parameters.
That is your *"components that must EARN their parameters or be deleted"* working
as written, and it is the first verdict in five weeks that was about Jack rather
than about the apparatus.
