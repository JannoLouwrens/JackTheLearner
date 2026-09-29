# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-29 06:37–07:1x UTC — the 130th audit.** Six hours after the 129th
(00:37–01:0x). The window is the builder's slots `01:07` through `06:07` and
demonstrated moved **107 → 107**.

Instrument exit codes, re-derived this sitting from source and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run status` **2**, `run review-queue` **2**.

Ratchet delta, quoted as a DELTA before anything is recorded, from `run status`'s
own SLOT LINE: **5 MOVED** (`fail_unowned_owned_forms` queue-row 29 → 30,
`review_queue_net_arrivals` 26 → **31**, `review_queue_violation_forms`
`{OVERDUE:1}` → `{HOLD-ON-A-RESOLVED-BLOCKER:9, OVERDUE:1}`,
`review_queue_violations` 1 → **10**, `unreachable` 95 → 96); 1 day-rolled
(`cpu_foreclosed_now`, the clock); no counter refused to compute; floors
**3 ABOVE** (`decisions_default_action_expired`, `pass_on_dead_dependency`,
`unreachable`), 0 BELOW, 0 UNVERIFIED.

**DISCLOSURE — the Review DAILY is sitting CONCURRENTLY and its numbers moved
under me mid-audit.** The collision is `37 */6` against `37 6`, 100 % on every
morning sitting. It committed `6d57068` (*"lg12 RULED on its fifth date"*) while
I was reading, and at the time of writing the tree carries `M
docs/REVIEW_QUEUE.md`, `M experiments/cpu_budget.json`, `M
experiments/ledger.json` — a spec is in flight from that desk right now. I
therefore quote **every** reading rather than one. At my sitting's open the
queue read **63 OPEN / 3 HELD / 18 DISPOSITIONED of 113 routed**, `15 = 6
OVERDUE + 9 HOLD`. Mid-audit: **61 / 3 / 20**, `13 = 4 + 9`. On the re-run
immediately before this commit: **59 OPEN / 3 HELD / 23 DISPOSITIONED / 28
ACTED / 1 DECLINED of 114 routed**, `10 = **1** OVERDUE + 9 HOLD` — the desk
has taken the OVERDUE class from 6 to 1 in the forty minutes I have been
reading, across at least four numbered acts (`6d57068`, `418c475`, `e41c932`,
…). **The numbers below are the last ones, and they were falling while I wrote
them.** I committed `docs/OVERSIGHT.md` and `docs/LESSONS.md` **by name**; both
files I touched (`OVERSIGHT.md`, `LESSONS.md`) are in `protocol.PROSE_DOCS`,
verified at HEAD, so this draft cannot dirty-stamp the desk's run.

**What I did NOT do, named rather than omitted.** I re-ran no spec (the
no-dry-run rule). I armed no decision: `decisions_undeclared` reads **0** —
there is nothing armable on the register — and manufacturing an entry to satisfy
the per-audit arming quota is the disease the quota exists to prevent. I
appended nothing to `DECISIONS_NEEDED.md`: nothing I found needs the owner that
is not already routed, and an append between `:37` and the `:43` regate sweep
would dirty-stamp `T0.28` for nothing (129th audit, RANK 3).

---

## VERDICT: DRIFTING — the ledger's arithmetic is clean, I found no loosening, and the builder did nothing wrong. What I found is that the project's one standing red on ledger-amendment integrity has been quoted as "the same three, known and dispositioned" by five consecutive audits while **one of the three quietly changed identity**; and that my own predecessor's conduct order has turned into a self-sustaining unit of work whose subject is the project's clerical cron. Neither breaks a certificate. Both are invisible from inside a single slot.

---

## RANK 1 — `T0.27`'s `live_violations = 3` has not moved since 2026-09-02, and **the set behind it turned over**: `T0.17` left, a second `T0.29` violation dated 2026-09-06 entered, and it is named nowhere in this repository. Its two companion numbers moved too, unwatched: `checked_pairs` 8 → 13, `unauditable_pairs` 24 → 22. (MEDIUM-HIGH — integrity instrument, new this window)

### The measurement

`T0.27` is the executable form of overseer B2 (2026-08-13): every amendment of
an adverse verdict must be auditable by someone who is not its author. It is a
**DELIBERATELY-RED GATE** and `run status` prints exactly one scalar about it:

    T0.27  live_violations = 3  (unchanged since 2026-09-28T10:25:29)

I ran `audit_supersedes_fail` against the live ledger rather than reading the
scalar. The three are:

| spec | adverse row | stamp | preserved bytes |
|---|---|---|---|
| `LG.00` | VOID 2026-08-30T18:47:59 | `8faff43+dirty` | `refs/jack/failimpl/LG.00/2026-08-30T18-47-59` |
| `T0.29` | FAIL 2026-09-02T09:18:06 | `661a48f+dirty` | `refs/jack/failimpl/T0.29/2026-09-02T09-18-06` |
| `T0.29` | FAIL **2026-09-06T07:16:03** | `44e54a7+dirty` | `refs/jack/failimpl/T0.29/2026-09-06T07-16-03` |

The owning queue row — `t027-preserved-failimpl-as-artifact`, **ACTED
2026-09-07 in `0e60ac1`**, `REVIEW_QUEUE.md:2488` — records its last data point
on 2026-09-02 and names a *different* three:

> *"of the two live `T0.27` violations one is recoverable and one (`T0.17`) is
> not"* … *"the live count is now **3 violations, 8 checked pairs, 24
> unauditable** — the third is `T0.29` FAIL `661a48f+dirty`"*

**`T0.17` carries no adverse row at all today.** Its full 21-attempt history
holds zero FAIL and zero VOID entries; its only `+dirty` stamps
(`406f86f+dirty`, `9e847cf+dirty`, both 2026-09-04) are PASSes, and PASS is not
an audited source. It contributes nothing to the class. **The third slot it
vacated was filled four days after the row's last data point**, and the filler
has never been written down.

### Why this is the finding and not bookkeeping

Nothing here is dishonest and no certificate is wrong — I checked the new member
on its merits and it is the same benign class as the other two: the failing
bytes of the 2026-09-06 `T0.29` run **are** preserved and hash-verified under
`refs/jack/failimpl/`, so it is a violation of the *"stamped at a committed tree
state"* clause and nothing worse. **The defect is that `3` was read as a
constant by five consecutive audits — the 126th, 127th, 128th, 129th and my own
draft — each of which recorded it as "known, owned, dispositioned", when the
disposition on file discharges a membership that no longer holds.** A terminal
`ACTED` row is never re-read (the 100th audit's own rule), so the description
froze on 09-02 while the set kept moving. This is the
`PASS-ON-DEAD-DEPENDENCY` disease one level up: there the board rendered a
*stored status* nothing re-evaluated; here it renders a *stored count* whose
contents nothing re-evaluated.

### And the mechanism that produced it is recurring, not historical

Both `T0.29` violations were created by **an organ running a spec out of its own
uncommitted working tree in the middle of its own sitting.** The 09-02 one the
queue row already attributes to *"the 61st audit's own B4 work"*. The unnamed
09-06 one is stamped `44e54a7+dirty` at **07:16:03**, four minutes after
`44e54a7` — *"PROGRESS_LOG: the 2026-09-06 FULL line"* — was committed at
07:12:55 by the Review's Sunday FULL. That is two desks, four days apart, in the
same trap, and it is the trap `docs/LESSONS.md` already calls *the dirty stamp
is tree-wide*. **It is also live right now**: as I write, `experiments/ledger.json`
is modified in a tree that also carries `M docs/REVIEW_QUEUE.md`.

### The verified negative, recorded so nobody re-derives it

I dated every one of the **22 unauditable pairs**. All 22 fall between
**2026-08-09T13:49 and 2026-08-13T02:34** — every single one is a pair where the
adverse side predates `impl_sha`, exactly as `audit_supersedes_fail`'s docstring
claims (*"absence is a historical gap, never evidence of dishonesty"*). **The
claim is true and the class is closed.** Latest member: `T2.08`, 2026-08-13 —
the original B2 fixture.

But it is closed **by accident of history, not by construction**. The function
decides `unauditable` on *field absence* (`if not e.get("impl_sha") or not
nxt.get("impl_sha")`), never on a date; and `live_unauditable_pairs` reaches no
ratchet key (I checked all 22 keys in `ratchet_readings.json`), no floor, and no
exit code. A row written today without an `impl_sha` would join the historical
gap silently and shrink the auditor's own coverage with nothing printing. The
repair is free because the number can only shrink from here — see FOR THE
BUILDER 2.

---

## RANK 2 — the 129th audit's conduct order has become a **work generator**. Five of six slots in this window produced a journal entry and nothing else; the regate forecast chain is the lead item of four of the six commit messages, and twice in this window its subject was a cron that did literally nothing. The chain has no termination condition and each slot holds only one link of it. (MEDIUM — drift; the builder is following an order correctly and this is my organ's own doing)

### What the six slots produced

| slot | commit | diffstat |
|---|---|---|
| `01:07` | `c23abd4` | `experiments/ratchet_readings.json` — **the 129th's FTB 2, discharged** |
| `02:07` | `2ae2775` | `docs/LOOP_JOURNAL.md` +15, nothing else |
| `03:07` | `012e935` | `docs/LOOP_JOURNAL.md` +15, nothing else |
| `04:07` | `e9f6d11` | `docs/LOOP_JOURNAL.md` +15, nothing else |
| `05:07` | `2b26ba9` | `docs/LOOP_JOURNAL.md` +15, nothing else |
| `06:07` | `311304d` | `docs/LOOP_JOURNAL.md` +14, nothing else |

Six `rc=0`, zero dark slots, zero ledger rows, zero verdict changes, **74 lines
of journal**, ~2 points of `week:all models` (31 % → 33 %).

### The loop, and the arithmetic that shows it is closed

My predecessor's FTB 1(a) said: *"when the unit is 'observe a scheduled external
event', the correct act is to pre-register the check for the next slot and
end."* That was the right repair for the `20:07` slot, which tried to *wait* for
a `:43` tick with a ~7-minute lifetime. **The builder adopted it exactly and
without exception.** The emergent consequence is that every slot now inherits a
forecast to close and owes a forecast to the next one, so no slot is ever
empty-handed — and the subject of the chain is `regate.sh`, the clerical lane
that re-buys stale documentation certificates.

From `/data/jack-logs/regate.log`, verified at source rather than from the
journal:

    2026-09-29T00:43  re-bought 2 of 2  (T0.21 PASS 9.98s, T0.28 PASS 46.76s)
    2026-09-29T02:43  nothing cheap is stale — no runner-output change
    2026-09-29T04:43  nothing cheap is stale — no runner-output change

The `03:0x` and `05:0x` slots each spent their headline unit confirming that a
cron which did nothing did nothing, and the `05:0x` commit message reports it
as **"third consecutive exact closure."** The `04:0x` slot, with no tick to
close, verified the *basis* of a forecast about the next one. That is three
consecutive slots whose lead line is about the project's own paperwork sweep.

### Why it matters, stated carefully

**This is not idleness and I am not accusing the builder of manufacturing
work.** I re-derived the board myself rather than taking it: `run next` reads
**0 fresh of 51** (37 carrying a settled verdict, 14 held) — the sixteenth
consecutive verified-empty board — and every slot in this window *also* did one
small, distinct, non-repeating verification (the legality-reader ownership
reading; the `T6.03` refusal mislabel corrected from `dirty-stamp` to the log's
actual *"blocked by T2.10 FAIL"*; today's 7-row due pile attributed to the desk
rather than to builder debt; the `ps09` oracle-cut hold re-read at
`REVIEW_QUEUE.md:11347`). Those are honest and worth having.

The defect is in the **headline**, and it is the same failure mode the 129th
named one rank above: *it reads as diligence*. A reader scanning six commit
titles sees forecasts made and closed exactly, three times running. What
actually happened is 107 → 107. No sentence in `GOAL.md` is served by predicting
the behaviour of `regate.sh`; the fourth clause of the first principle covers
*protecting the honesty of watching what happens when brain, body and world
meet*, and this is the machine watching its own filing cabinet. **A forecast
whose answer is "nothing will happen", made about a tool that returned "nothing
cheap is stale" on its last two ticks, is not worth a slot's headline and is
certainly not worth re-arming.** One line of conduct — FOR THE BUILDER 1.

---

## RANK 3 — the window served no `GOAL.md` sentence about Jack, the creature gate is still NONE, and the 7-day picture is worse than the 6-hour one. (MEDIUM — drift, and the builder is not its cause)

**Which `GOAL.md` sentence each slot serves**, per the mandate:

- `01:07` — ratchet-note discharge. First principle, fourth clause (*protects the
  honesty of watching*). Instrument work.
- `02:07`–`06:07` — board re-derivation (mandated), forecast chain (RANK 2), and
  one small verification each. Instrument work. **None is about Jack.**

**Over seven days, from `run status`'s own SETTLE EVENTS**, which I read rather
than inherited: **137 runs recorded = 4 first-ever verdicts, 123 re-buys, 10
status changes.** **110 of 137 (80 %) are instrument-coupled** — the spec
declares a measurement-machinery file in `IMPL_DEPS`, so this project's own tool
edits are what staled the certificate. Of the four first-ever verdicts, two
(`T0.21`, `T0.31`) are the rig proving the rig; the re-buy roll-up is led by
`T0.21`×20 and `T0.31`×20. **In a week of 137 runs, the number that settled
something new about the creature is two** (`T4.06` PASS, `LT.03` VOID).

**Creature gate: NONE**, recorded as the violation it is, and re-derived rather
than quoted: `T6.01 ← T4.05 ← T4.04 ← T2.01 ← T1.08 (FAIL)`. `T1.08`'s
pipeline-repair design is the Review's, `DUE 2026-10-02`.

---

## The mandated sections, with the findings above not repeated

**1. Integrity of the ledger — CLEAN, re-derived not inherited.** All **107**
standing PASS rows carry a `commit` field (**0 empty**); the **57** distinct
commits behind them all still exist in git (`git cat-file -e` over the distinct
set, **0 missing**). The control half, which my predecessor did not run:
`experiments/verify` **EXIT 0** — `declared_control_never_ran` is at its pinned
1-of-1 probe value and `no_control_specs` reports honestly; no PASS in this
ledger declares a control that its test never called. Standing classes, reported
honestly by `run status`: **2 DIRTY STAMPS** (`T6.03`, `PL.02` — unchanged, both
refused with written reasons), **15 STALE CLAIMS** + 1 pre-`impl_sha` (`T2.02`),
**5 UNBACKED CERTIFICATES** (legal, reporting-only), **1 DELIBERATELY-RED GATE**
(`T0.27` — RANK 1). Ledger totals: `PASS 107 / FAIL 32 / VOID 16 / BLOCKED 1` =
156 rows with a verdict against **254 registered** → **42.1 %**.

**2. Thresholds and controls over seven days — NO SILENT LOOSENING FOUND.**
In my own window this is trivial and I verified it rather than assumed it: the
six slots touched **two files total** — `docs/LOOP_JOURNAL.md` and
`experiments/ratchet_readings.json`. No spec, no test, no registry file was
edited by anyone in six hours. Over the full seven days I re-ran the loosening
grep (`_MIN`/`_MAX`/`_FLOOR`/`_CEIL`/seed counts/comparison operators/` or `)
across `registry.py`, `registry_expansion.py` and `experiments/tests/`, and
**I deliberately spot-checked a spec neither of my two predecessors used** so
that three audits are not one check counted three times: `LT.02`'s attempt-2
redesign (`88762a2`), which is the riskiest shape in the window — a *redesign*
that re-scopes a gate onto a newly-added arm, where a bar can be "unmoved" in
source and weaker in meaning. It holds. `OCC_ICM_MIN = 3.0` and
`CONTROL_OCC_MIN = 3.0` are byte-unchanged at `lt_02_chaos_detector.py:308,312`;
the only numeric constant the commit adds is `ACT_NOISE_SIGMA = 1.0`, the new
arm's own parameter; the four controls that passed attempt 1 are all carried
unchanged, and the commit adds two new gates (`C1b`, `V5`). The redesign did not
buy a PASS either — `LT.02` is FAIL today, and the one PASS it did buy on 09-25
was caught as an epsilon artifact by the builder itself. **No findings in
section 2.** Stated plainly because it is true.

**3. Drift from the goal.** Covered in RANK 3. The converse, from `coverage`
EXIT 2: **0 commitments with NO declared spec** (floor held), **3 CLAIM-DEAD**
(smell, shelter/building, thermal-kills) and **14 more with live claim specs and
nothing passing** — **17 of the owner's constitutional commitments with zero
passing claims**, including *too cold kills him*, *he builds a shelter*, touch,
tool use, proprioception, sleep, plasticity, fast/slow and the told world.
`NO-LIVE-PATH` stands at **6** distinct commitments/seats; the repair for every
one is a **registration**, never an unpark and never a deletion.
`goal_unrunnable = 7`, unchanged since 09-05, with `GEN.02`/`GEN.03`/`GEN.06`/
`GEN.09` flagged **4 NEW unrunnable citations** — owned by the OPEN row
`gen-four-reparented-to-a-decision-that-had-already-closed` (DUE 10-01).
`GOAL.md` cites 16 spec ids, **0 dangling**.

**4. Builder liveness.** 6 iteration starts (`01:07`–`06:07`), **6 ended
`rc=0`**, 0 session-limit deaths, `lost_iterations.log` correctly 0 bytes,
`dark_slots` 0. PASS delta **107 → 107**. `run next` read **0 fresh of 51** on
every slot, and I re-derived it myself at 07:0x rather than inheriting it: the
sixteenth consecutive verified-empty board. **Nothing was manufactured**, and
that is now the eighth consecutive audit at which I can say so. The 129th's
RANK 1 (a slot waiting for an event outside its lifetime) is **discharged**: no
slot in this window armed a wait, and every one pre-registered and ended. Its
FTB 2 is discharged at `c23abd4` and I verified the note at source, not from the
commit message — `unreachable`'s key now carries its cause and
`UNREACHABLE_BASELINE` correctly still reads **95**.

**5. Compute honesty.** Re-derived from `gpu_budget.json`'s own per-week records
rather than a summary: **`2026-W39` does not appear in the `weeks` map at all —
zero jobs, 0.00 h spent — against 30.0 free Kaggle GPU-hours expiring Saturday
2026-10-03.** After `W37` (5.25 h: colab 3.033 + kaggle 1.379 + failed 0.837)
and `W38` (0.92 h), this is the **third consecutive week substantially lost**.
Every cost class in `coverage` reads `NOT FILLABLE`; **3 are empty with no path
in** (`cpu<1min`, `gpu<20min`, `gpu<8h`). Both live routes run through `T1.08`
(FAIL, blocks 45), whose repair design is undesigned until 10-02 — **one day
before the hours expire**. **No dispatch has been manufactured and none should
be.** Standing waste unchanged: `gpu_hours_no_verdict` TOTAL **48.42 h**, of
which `D1.0` alone holds **33.78 h across 2 attempts and 0 verdicts**;
`gpu_unattributed_jobs = 21`, AT floor.

**6. Stuck decisions.** `decisions --check` EXIT 1. **No `MEANS-ESCALATED`** —
nothing a measurement could settle sits on the owner's desk; the `D1` disease is
absent. **`decisions_undeclared` = 0, at floor** — nothing was armable and I
armed nothing rather than invent an entry. One armed: `D37`, due 10-04, default
(iii) HOLD, `costs 0 specs` and the entry says so itself. Five not armed: `D33`
(stale 6 d), `D35` (stale 5 d), `D38` — all `CONDUCT-DESK`; `D33` additionally
`DEFAULT-ACTION-EXPIRED` (the one class above floor, 1 vs 0); `D37` flagged
`CONDUCT-MISFILED?`. `decisions_unrouted_owner_ask` and
`decisions_vanished_owner_ask` both **0**, at floor — the `D15` disease this
organ measured on 08-29 stays closed, and `PROGRESS.md`'s single owner-ask is
correctly attributed to `D22`/`D33`. `decisions_firing_diff = 0`: no owner
decision was quietly acted on without being recorded.

**7. Bakeoff hygiene — `champions --check` EXIT 0, 10 violations, all standing
and none new.** **Learning core** held `BY VERDICT` off `LC.03` (a VOID) with
`VERDICT-IS-A-VOID` **and** `TRIGGER-UNREACHABLE` — that is this section's *"a
VOID treated as a verdict"*, printed rather than hidden, and `D37` is the live
entry on it. Also standing: **World** `VERDICT-UNDECLARED`/`TRIGGER-UNDECLARED`;
**Fast/slow coupling** `ARENA-UNREACHABLE`/`TRIGGER-UNREACHABLE`; 2 `NO-ARENA`
(ASR, Speaker ID); 2 `UNCONTESTED` (Vision encoder, PLASTIC ONLY).
`champions_unwinnable = 4` AT floor, `champions_trigger_debt = 3`.
**`ARENA-MISSING` remains 0** — my own standing prompt still says 8, and those
seats were repaired by REGISTERING specs, never by deleting arena references.
On *"a winner chosen inside the noise margin"*: `run status`'s ANCHOR-DECIDED
block still reports `T4.06`'s `loss_reweight` certified at **+0.0187 = 6.9 % of
the incumbent's own 0.2699 seed spread with one seed of three regressing**. That
is disclosed on the certificate (`f7900b5`) and the adoption is the Review's, so
it is not mine to rule — but it is the weakest "winner" on the board and the
field watch's week-9 §6 attacks the same statistic from the other side. **§6 is
still `UNROUTED-FIELD-FINDING` after the builder independently verified every
number in it** (`3e5ac1e`); the draft it lives in is sealed `rc=124` and its
consumption is the Review's deferred act. Correctly not routed by either of us;
recorded here so the verification is not lost if the draft is rewritten.

**8. THE HONEST SUMMARY — are we closer to a curious humanoid that climbs the
ladder than we were six hours ago?**

No, and for the second audit running the reason is not that anyone did anything
wrong.

Six slots, six `rc=0`, one FTB item discharged and verified at source, one
predecessor's mislabel corrected, four forecasts closed exactly, zero waits
armed, zero specs re-run to fill a slot, zero bars moved, zero manufactured
work. The builder is in good health and has been for a fortnight. **And 107 →
107, no spec about Jack ran, and five of the six slots left behind nothing but a
paragraph.**

What this audit adds to that picture is two things about *seeing*, which is what
this organ is for. The first is that a number can stand still while the thing it
counts changes underneath it — `T0.27` has read `3` for twenty-seven days and
one of the three is not the one the disposition discharged. Five audits,
including the two before mine, quoted it as settled. The second is that when a
board is empty, work will invent itself in whatever shape the last order left
open: my predecessor told the builder to pre-register instead of wait, the
builder obeyed perfectly, and the result is a chain of forecasts about a filing
cabinet that closes exactly every time and will never stop.

Set against the ladder, nothing moved: seventeen of the owner's constitutional
commitments still have no passing claim. Nine live rows still read
`HOLD-ON-A-RESOLVED-BLOCKER` behind a world whose authorship was formally
DECLINED three days ago and has no owner. Thirty free GPU-hours expire on
Saturday for the third consecutive week, one day after the design that could
spend them is due. `T1.08` — one red Tier-1 spec — blocks forty-five others
including every creature gate and every Tier-5 claim. In seven days and 137
runs, two verdicts were about the creature. Of 31 resolved decisions, 30 were
resolved by armed default and **one, ever, by the owner** (`D19`, 2026-09-17).

**We are not closer to Jack. The one encouraging fact in this window is not on
the ladder at all: the Review desk is, as I write, disposing its OVERDUE class
row by row — `6 → 4 → 1` in the forty minutes of this audit, `lg12` ruled on
its fifth date and the `so10`/`lg13` seat-race pair ruled together as its own
page promised. That desk's design capacity, not the builder's throughput, is
the measured bottleneck, and this morning it is the only thing in the project
that is moving.** Whether that pace survives contact with `T1.08`'s design —
the one unit that would refill the builder's board, due 10-02 — is the
question the next audit should open with.

---

## FOR THE BUILDER

1. **Stop re-arming the regate forecast chain, and stop leading a commit
   message with its closure.** RANK 2. This is conduct, not code — no counter,
   no queue row, no `D35` question. The 129th's order was *"pre-register instead
   of wait"*, and you executed it correctly; what it did not intend is a
   perpetual unit. Concretely: **pre-register a `:43` tick only when the
   forecast is non-trivial** — i.e. when `run status` shows something cheap and
   stale that the sweep could actually buy. When the last tick logged *"nothing
   cheap is stale"* and nothing has re-staled since (which you already check),
   the honest journal line is **"no forecast owed; nothing is stale"**, one
   sentence, and the slot's headline goes to whatever else it did. `04:43` and
   `02:43` both read *"nothing cheap is stale — no runner-output change"*, and
   two of your six commit messages lead on confirming it. Keep verifying the
   ticks that admit; they are real. Do not build anything.

2. **Floor `live_unauditable_pairs` at 22, shrink-only, in the same idiom as
   `PASS_ON_DEAD_DEPENDENCY_BASELINE`.** RANK 1's verified negative. I dated all
   22 pairs: every one falls between `2026-08-09T13:49` and `2026-08-13T02:34`,
   so the class is a genuinely closed historical gap and the constant is free
   today. But `audit_supersedes_fail` (`experiments/protocol.py:3355`) decides
   the class on **field absence** (`if not e.get("impl_sha") or not
   nxt.get("impl_sha")`), never on a date, and the number reaches **no ratchet
   key** (I checked all 22 in `ratchet_readings.json`), no floor and no exit
   code. A future row written without an `impl_sha` would silently shrink
   `T0.27`'s own coverage. Floor it where it stands. Do **not** touch the
   `+dirty` gate or `live_violations` — that is `D16`, the owner's, and `T0.27`
   is deliberately red.

3. **Nothing else.** Item 1 costs a line of conduct; item 2 costs one constant
   and one ratchet key. Neither touches a bar, a seed, a control or the ledger,
   and the queue is running 6.00 arrivals per cycle against 1.71 disposals —
   this is not the week to add rows.

## FOR THE OWNER

**1. `D33` — who authors the W1 world edit? Six days past its deadline, and
neither organ can clear the red.** Unchanged from the 126th through 129th
audits, repeated because nine live rows are behind it and it remains the largest
single fact about this project. `decisions_default_action_expired` reads **1**
against floor **0**, unchanged since 2026-09-23. `D33`'s own default is *"re-date
once more, to 2026-09-23"* — an act in the past on the earliest day it could
fire. The Review has formally **DECLINED** the W1 authorship (the first
`DECLINED` in 113 routed rows) and correctly refuses to move `decide_by`; `D13`
bars me from editing the register's rulings. The instrument names two legal
repairs — **shorten `decide_by`** (a deadline may tighten, never lengthen) or
**declare whose date it is with `(CLOCK: <whose>)`** — and **neither is
available to either organ.** Nine live rows read `HOLD-ON-A-RESOLVED-BLOCKER`,
two of them (`ne01-occlusion-knife-edge`, `water-apply-phantom-force`) **36 days
old with no `DUE:` at all**. The Review recommends option (ii) — the builder
drafts under Review — and says it may not carve that exception out of `D22`,
which is your ruling. A stop-rule fires 2026-10-09.

**2. PERISHABLE — 30.0 free Kaggle GPU-hours expire Saturday 2026-10-03, the
design that could spend them is due 2026-10-02, and this would be the third
consecutive week lost.** NO-DECISION; re-derived from `gpu_budget.json`'s own
per-week records (`W37` 5.25 h, `W38` 0.92 h, **`W39` has no entry at all**).
Every cost class reads `NOT FILLABLE` and three are empty with no path in. Both
live routes run through `T1.08` (FAIL, blocks 45), whose repair design is the
Review's. **No dispatch has been manufactured and none should be** — the scarce
resource is a designed unblock and your ruling in item 1, not a machine hour.
Standing waste unchanged: `D1.0` holds **33.78 GPU-hours across 2 attempts and
0 verdicts**.

**3. PERISHABLE — your headline number is still scheduled to fall by one on
2026-10-04, by cron, and it is an artefact.** `T0.28` requires at least one
*armed* decision to exist in `DECISIONS_NEEDED.md` and currently passes on
`live_armed = 1.0` from the single entry `D37`, whose `decide_by` is 2026-10-04.
**Ruling `D37` will subtract one from the demonstrated count within about two
hours.** That is the certificate's shape, not a loss. Do not let it discourage
the ruling; the repair is a Review disposition and the forecast is written on
the queue row.

**4. `D35`'s three one-line repairs, unchanged from the 116th, 125th, 126th,
127th, 128th and 129th audits.** (a) a reachable release condition — `T6.01`
sits behind `T4.05 ← T4.04 ← T2.01 ← T1.08` (FAIL), so the freeze cannot end by
any act available to anyone; (b) a clause-2 exemption for truthfulness repairs
and floors on EXISTING checkers, which would unblock the unfloored
`no_control_specs` — **and FOR THE BUILDER 2 above is a second instance of
exactly that class, so this clause is now blocking two repairs, not one**;
(c) confirm the freeze is meant to be unbounded; (d) say whether clause 2's
*"audit organ, checker or ratchet"* covers autonomous **actors**, since it
stopped a one-expression read-only join and did not reach a cron lane that
writes the ledger.

**5. NO-DECISION, liveness: 0 dark slots, 6 of 6 builder slots `rc=0`, and the
Review desk is sitting as this is written.** All four organs fired within
cadence. The builder has now verified an empty board sixteen consecutive times
without inventing work — correct behaviour, and also the measurement that the
bottleneck is the two design debts in items 1 and 2, both of which sit at a desk
that sits for twenty minutes a day. **That desk took its OVERDUE class from 6
to 1 during this audit** (`review_queue_violations` 15 → 13 → 10 across my three
re-runs), which is the only forward motion in the window and the best sitting it
has had in a fortnight.
