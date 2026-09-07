# OVERSIGHT — 80th audit, 2026-09-07 00:37–01:0x UTC (at `16a8194`, tree clean)

## VERDICT: DRIFTING — **the ledger is clean and nothing was loosened; but the field written to close a named provenance hole records the wrong value, and the run about to land four of them is the most expensive spec this project owns.**

Section 1 is clean: 106 PASS rows, 0 dead commits, 0 missing implementations,
0 missing controls, 2 argued exemptions. Section 2 is clean and I checked the
highest-risk change in the window myself rather than taking the Review's word:
the adopted `D1.0` gate was committed **44 seconds** before a ~17.6 GPU-hour
dispatch, and it is a genuine tightening (§2.1). No threshold moved in the
loosening direction anywhere in seven days.

The finding is §1.1. `D1.0`'s ledger row records a `head` per kernel — added by
the 59th audit B7 to close the scar `LESSONS.md` states as *"a row naming only
the local commit asserts a provenance nobody checked"*. The code calls
`_head_sha()` **after** each kernel returns, so it stamps the local HEAD at
**harvest** time, not the HEAD the kernel was **built** from. Those differ by
4–6 hours of hourly builder commits. All four of attempt 2's kernels will
record a commit they demonstrably did not run — and the correct value is
already written, one function away, in `gpu_submissions.jsonl`.

No PASS certificate is wrong today. The field has never been exercised:
attempt 1's row predates it. Attempt 2 harvests in the next few hours and is
the first row to carry it.

Ranked by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger — CLEAN, with one field about to become false

Mechanical sweep over all **106 PASS** rows of **245** registered (143 rows total):

- **0** PASS rows whose recorded `commit` fails `git cat-file -e <sha>^{commit}`.
- **0** PASS rows with no spec in `BY_ID`; **0** with no implementation module.
- **0** PASS rows whose spec declares no `control`.
- **104 / 106** carry populated `control_metrics`. The two that do not are
  `T0.01` and `T0.10`, both declaring `"NONE, BY DECISION (52nd audit B5)"`
  with the reasoning in the spec. Argued exemptions, not silence.
  **No PASS in this ledger is a claim without evidence.**

### 1.1 `D1.0` stamps each kernel with the HEAD it HARVESTED at, not the HEAD it RAN — and every one of attempt 2's four kernels will be wrong

`experiments/tests/d1_0_control_path_bakeoff.py:850-857` carries this comment:

> `head` per kernel (59th audit B7): the row's top-level `commit` is stamped
> once at run start, but a multi-kernel dispatch can span pushes — **each
> kernel names the HEAD it was actually built from.**

It does not. `submit(...)` **blocks** until the kernel finishes; `_head_sha()`
is called on the line after it returns. The value stamped is the local HEAD at
the moment that kernel's *result was collected*. The builder loop commits
roughly hourly and each of these kernels runs 1.5–6 hours, so the two are
systematically different.

This is not a hypothetical. Both halves of the join are on disk —
`experiments/gpu_submissions.jsonl` records the dispatch head, `git log`
records what HEAD was at each finish:

| kernel | pushed | head it RAN | finished | head it will RECORD | off by |
|---|---|---|---|---|---|
| G3 ref `…1788682804` | 08:20:04 | `3a4ccfd` | 09:52:37 | `1ee4a3a` (09:15:33) | 3 commits |
| `aprime`+`d_mlp` `…1788688360` | 09:52:40 | `1ee4a3a` | 13:57:09 | `321d786` (13:23:13) | 11 commits |
| `b_split` `…1788703032` | 13:57:12 | `321d786` | 19:57:38 | `ce2b56b` (19:14:00) | 10 commits |
| `c_e2e` `…1788724660` | 19:57:40 | `ce2b56b` | in flight (~01:5x–04:3x) | HEAD at harvest | ≥5 already |

**Four of four wrong.** Verified in flight: `kaggle kernels status
jannolouwrens/jack-ladder-1788724660` → `RUNNING` at 00:5x; runner pids
1775588/1775608 alive at 16 h 19 m, polling.

**Why this is worse than having no field.** The four dispatch heads span
`3a4ccfd..ce2b56b` — **47 files, 7,098 insertions**. The `head` column is the
only thing in the ledger that would let a future auditor ask *"did the arms of
this bakeoff run the same code?"*. Populated with harvest-time values it
answers a question nobody asked, in the format of the question that matters,
and a future mid-run change that DID move an arm would be invisible to the
exact record built to catch it. This is the 59th audit's own B2 principle —
*an id that resolves to a corpse is a worse dangling reference than one that
resolves to nothing* — landing on its own repair.

**I checked whether harm has already occurred, and it has not.**
`d1_0_control_path_bakeoff.py` does not appear in `git diff --name-only
3a4ccfd ce2b56b`, nor in attempt 1's span `9494cd1..566f840`. The arms of both
attempts ran identical spec code. The defect is in the *audit trail*, not
(this time) in the science.

**The correct value already exists.** `gpu.py:1084` computes `head =
_head_sha()` at dispatch and `_record_submission` writes it into
`gpu_submissions.jsonl`; `gpu.py:778` already surfaces it as `submitted_head`.
The repair is to read it back off the receipt instead of recomputing it.
Routed as **B1**.

### 1.2 Attempt 1's row records no `head` at all — the field's first-ever write is the one landing tonight

`ledger.json` `D1.0.metrics.kernels` holds `{arms, backend, gpu, wall_minutes}`
for three kernels and no `head` key: the 59th audit's fix (`112cf3b`,
09-01 20:21) landed *after* attempt 1 finished (18:23). So there is no prior
row to compare against and no way this was caught by re-reading one. It could
only be caught by reading the code against the clock, which is what §1.1 did.

---

## 2. Thresholds and controls, over time — CLEAN

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`. **No numeric threshold
moved in the loosening direction. No control was deleted or weakened. No
`_check` gained an `or`. No seed count was reduced. No assertion was removed.**

### 2.1 I re-derived the week's highest-risk change rather than accepting it

`8f2990d` (09-06 08:19:03) rewrote `D1.0`'s learning gate and the dispatch went
out at **08:20:04 — 61 seconds later**, committing ~17.6 GPU-hours. A gate
edited immediately before the most expensive run in the project is exactly
where a loosening would hide. The Review claims *"strictly harder"* and
*"3.0σ unmoved"*. Both check out:

- `MIN_LEARN_SIGMA = 3.0` — **unmoved**, byte for byte.
- The **statistic** changed, and in the hard direction: `sigma_vs_random` →
  `paired_t`, each arm against **its own untrained twin, paired by seed**.
  Attempt 1's twins read **2.94–2.96σ against random** against a 3.0 bar — the
  old null was one twentieth of a sigma from clearing the gate by architecture
  alone. Scoring against the twin removes that.
- Two **new required** conjuncts: `CONSISTENCY_MAX = 0.50` (G2) and
  `SB3_REFERENCE_FLOOR = 450.0` (G3, a verbatim external reference in its own
  1.8 h kernel that runs *first* and VOIDs the harness before the 15 arm-hours
  are spent).

Additive and harder on every axis. **Ordering respected too** — PROGRESS's
FOR-THE-BUILDER item 1 demanded the gate be committed *before* the dispatch,
not during it, and the 61-second gap is on the right side of that line.

The other constant that moved this week, `ME.9`'s `MIN_DISTRACTOR_EVAL` 9 → 12
(`4f62755`), is a **tightening** with the arithmetic recorded at the constant.

---

## 3. Drift from the goal

### 3.1 `coverage` EXIT 2 — the standing constitutional red, correctly owned

`0 commitment(s) with NO declared spec` — the 08-10 hole stays shut. The red is
**4 NEW unrunnable citations**: `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all
`welded<-LC.07` since `LC.07` went PILOT-BLOCKED. GOAL.md cites all four in the
present tense (the three expansions, §"jungle is the foundation") and every one
resolves to a corpse.

**This is the 79th audit's §3.4 and it is properly owned, not ignored.** I
checked the execution rather than the claim: FTB item 3 explicitly said the
page is the Review's to rewrite and the correction belongs on
`goal-cites-four-specs-that-resolve-to-corpses` (OPEN, DUE 09-10) — and that is
where the builder put it. `PROGRESS.md:171` still asserts the four are on
`GOAL_UNRUNNABLE_BASELINE` (`coverage.py:263` holds exactly `DP.02`, `DP.03`,
`LC.04`); that sentence stays false until the next FULL, by design, on a page
already sealed as a DRAFT. **No new finding — correctly routed.**

### 3.2 Four constitutional commitments are CLAIM-DEAD, and all four now have clocks

`smell`, `balance`, `shelter/building`, `thermal (kills)` — 0 passing claims,
every claim spec parked or foreclosed. Three of the owner's own words are in
that list (*"too cold kills him"*, *"he builds a shelter"*, every sense).
Unlike the 76th audit, which found these untouched, each now carries a dated
queue row: `sm03-heldout-split-saturated` (**DUE today, 09-07**),
`ba03-null-saturates-the-horizon` and `sh02-null-saturation` (DUE 09-09). The
repair in every case is a REDESIGN, not a dispatch. **Named, not a new finding
— but `sm03` is due today and the desk's measured capacity is 1 dated row per
cycle against 4 promised for 09-07.**

### 3.3 What the builder worked on, and what it serves

Last 24 h: the adopted `D1.0` gate (GOAL: *"architecture always contested"*);
`W1.00`/`W1.02` registered and run (GOAL: the world must be *consistent,
discoverable, consequential*); `EpisodicMemory`'s similarity floor and the
`ME.1`/`ME.9`/`XL.00` re-buys (GOAL: *"memory makes it him"*); `T0.35` and the
`IMPL_DEPS` sweep (GOAL: *"protects the honesty of watching"*); the `ORDERED:`
join. **Nothing served no GOAL.md sentence. No drift in the work itself.**

The drift is in the *shape*: ten consecutive slots (15:07 → 00:07) ended
`106 -> 106` while the one dispatchable venue was occupied. All ten were honest
about it and none invented a unit — but see §4.

---

## 4. Is the builder alive and productive? — alive, honest, and nearly stalled

**25 iterations** in the 24 h to 00:39, **25 ended `rc=0`**, zero aborts, zero
paused loops, `lost_iterations.log` still 0 bytes. Cadence exactly hourly.

**PASS delta: 105 → 106 (+1). Registered: 242 → 245 (+3).** The intraday path
was 105 → 104 (`ME.1` strengthened into a FAIL) → 107 → 106, then **flat at
106/245 for the last ten slots and 9.5 hours.**

The board is genuinely empty and the tool says so: `coverage`'s QUEUE DEPTH
reads **6 dispatchable, 6 VOID, 0 FRESH**, with every cost class marked
`NOT FILLABLE` — three empty outright, the rest blocked behind a pilot whose
repair is a redesign. The builder correctly refused to invent work.

**So this is not idleness, and I am not calling it that.** It is the thing
§3.2 describes seen from the other end: the ladder has no next rung that a
dispatch can buy, because five specs are PILOT-BLOCKED and the repairs are all
redesigns sitting on the Review's desk. One demonstrated capability in a day of
25 healthy iterations is the honest number, and the constraint is upstream of
the loop.

---

## 5. Compute honesty — accounted, and expensive

`gpu_budget.json` W36: **11.6203 kaggle-hours**, all three charged jobs
`ok: true`, every one reconciling exactly against `gpu_submissions.jsonl`
(1.5412 + 4.0735 + 6.0056). No overruns. No orphaned spend — every W36 job
carries a matching attempt receipt with `pid 1775608`.

**All 11.62 h are `D1.0` attempt 2, and a fourth kernel (est 8.625 h) is in
flight.** Attempt 2 will close at **~17.6 h**. Attempt 1 spent **16.17 h** and
returned a **VOID**. That is **~34 GPU-hours on one spec with no ledger verdict
yet** — a third of the free allocation for two calendar months of the project.

This is not waste and I want to be precise about why: it is the arbitration
that unblocks `T2.01` (frees 34 specs) and it is the only live arena the
architecture invariant has. The G3 reference kernel exists specifically to VOID
the harness for 1.8 h instead of 15 when the venue is broken. The spend is
argued, pre-registered and reconciled. **It is simply the largest single bet on
the board, and §1.1 is about the audit trail that bet is writing.**

Kaggle's 30 h resets Sunday; W36 closes at ~17.6 of 30 with the in-flight
kernel spanning the 04:59 reset.

---

## 6. Stuck decisions — CLEAN

`decisions --check` **EXIT 0**. `ratchet ok (0/10 undeclared, 0/3
unrouted-owner-ask, 0/0 vanished-owner-ask, 0/0 default-action-expired)`.

- **0 `MEANS-ESCALATED`** — no measurable fork is sitting on the owner's desk.
  The D1 disease is not present.
- **0 `UNDECLARED`** — nothing to arm this audit. The ratchet is at its floor;
  I am not arming an entry for the sake of the instruction when there is no
  undeclared entry to arm.
- **0 `OVERDUE`** — no default is due to fire.
- `D23` and `D24` both `decide_by` **2026-09-11**, both with legal defaults
  (measure-don't-gate; declare-don't-decide). Neither widens what is permitted.
- PROGRESS's FOR-THE-OWNER item 1 is attributed to `D24 (cites)`; items 2–4 are
  self-declared NO-DECISION. **Every owner-ask on that page reached a desk.**

## 7. Bakeoff hygiene — CLEAN

`champions --check` **EXIT 0**, every ratchet at or below baseline
(0/0 phantom arena, 2/3 unfalsifiable, 2+1/4 uncontestable, 2/2 unverified
verdicts, 3/3 trigger debt). Every seat says what would unseat it.

`DECISIONS_RESOLVED.md`: no decision made without a learning gate, no VOID
treated as a verdict, no winner chosen inside the noise margin. `D10`'s
wm-latent seat is seated BY VERDICT off `LC.03`'s VOID **with the single-arm
caveat recorded on its face** — the honest form. `D15`'s firing carries two
amendments made six hours later, both recorded as repairs of the fired code
rather than reversals, with the reversal path stated.

`run review-queue` **EXIT 0, 0 violations.** No `OVERDUE`, no `STALE`, no
hold without a clock. The one live violation (`t211-diayn`) was re-armed at
00:11 with a reason, which is the legal repair.

**Named, not a violation:** 10 rows share **2026-09-13** against a measured
consumer capacity of **1 dated row per cycle**, and 09-09 carries 7. The tool
prints this as amber and it is right to. Those promises are scheduled to break
together, and the next free date is 2026-09-17.

---

## 8. The honest summary

**Closer — by one rung, and the rung is real.** `W1.02` PASSed: outcomes in
Jack's world have resolution, which is a precondition for anything he could
learn by living. `ME.1` went FAIL because a control got honest enough to catch
his memory confabulating on 100% of the questions it should refuse — a
subtraction from the scoreboard that made the scoreboard worth more.

**But mostly closer to a longer list of green ticks.** Total ledger writes this
week ran ~219 against 78 the week before, and the ratio is the certificate
re-buy treadmill, not work on Jack. Ten of the last twenty-four hours moved
nothing. And the four commitments that are *most* about a creature living in a
world — smell, balance, shelter, cold that kills — have **zero passing claims
between them**, all four blocked on redesigns rather than compute.

The gap is not honesty and it is not effort. Every instrument in this repo is
green tonight and each was earned. The gap is that **W0 is the bottleneck and
five separate specs have now measured it from five directions** — `SH.01`'s
oracle that cannot seek, `SH.02`'s null that saturates the roof, `BA.03`'s
blind twin holding 98.9% of the horizon, `DP.04`'s lifespan with no resolution,
`SM.03`'s saturated split. That is not five bugs. That is one world that is too
shallow to falsify claims about living in it, and the `W1` family is the first
honest attempt to say so.

Jack can see, hear, speak, remember whom he spoke to, and die and come back
with his diary. He still cannot get cold, seek shelter, or fall over — and
tonight the reason is a world, not a brain.

---

## FOR THE BUILDER

1. **`D1.0` must record the HEAD each kernel RAN, not the HEAD it harvested at
   (§1.1) — and this is time-critical: kernel 4 harvests within hours.**
   `d1_0_control_path_bakeoff.py:827` and `:857` both call `_head_sha()` *after*
   the blocking `submit(...)` returns, stamping harvest-time HEAD. All four of
   attempt 2's kernels will name a commit they did not run (table in §1.1;
   off by 3, 11, 10 and ≥5 commits).
   **The correct value is already on disk** — `submit` writes `"head"` into
   `experiments/gpu_submissions.jsonl` at dispatch (`gpu.py:1128`) and
   `gpu.py:778` already reads it back as `submitted_head`. Have `submit` return
   the dispatch head on its `JobResult` (or read the last attempt receipt for
   the returned `attempt_id`) and stamp **that**.
   **Do not** fix this by deleting the `head` key — a missing field is honest
   and a wrong one is not, but the field is the only thing that could ever
   answer "did the arms run the same code?" and deleting it is the repair that
   lowers its own number.
   **If attempt 2 harvests before the fix lands, commit the row as found and
   amend it** with the four dispatch heads from `gpu_submissions.jsonl`, with
   the reason recorded — do not re-run anything, and do not hand-edit the
   metrics silently.
2. **Correct the comment in the same commit.** Lines 850–852 assert *"each
   kernel names the HEAD it was actually built from"*. A comment that states
   the property the code fails to have is how this survived the 59th audit's
   own review; the next reader must not be told it is solved.
3. **Check the same call pattern anywhere else a multi-kernel spec merges
   results.** `_head_sha()` has exactly three call sites today
   (`gpu.py:1084`, correct — at dispatch; and the two in `d1_0`, both wrong).
   That is a small enough set to settle permanently: any `_head_sha()` called
   after a `submit()` returns is this bug. Consider a `T0.*` property asserting
   it, in the idiom `T0.35` already established for undeclared imports.
4. **`sm03-heldout-split-saturated` is DUE today (09-07)** and three other rows
   share the date against a measured capacity of 1. Either act on it, decline
   it, or re-arm it with a new `DUE:` and a reason — the three honest repairs.
   It is one of the four CLAIM-DEAD commitments (§3.2) and `smell` has had no
   live claim since 2026-08-20.
5. **Standing, unchanged:** `D1.0` may not be re-dispatched outside the adopted
   gate; `HR.1`–`HR.4` stay D19-held to 09-14; `LF.01` attempt 2 waits for the
   09-09 design; `T0.11`'s re-buy (oldest live certificate, 34 days) unblocks
   the moment the venue frees.

---

## FOR THE OWNER

**1. NO-DECISION — a report, already routed to the builder.** The most
expensive measurement this project has ever run is writing a provenance record
that is wrong in every row. `D1.0` has cost ~34 GPU-hours across two attempts;
attempt 1 returned VOID; attempt 2 lands tonight. Its ledger row will name four
commits that its four kernels did not run (§1.1). **No existing certificate is
affected and no science is invalidated** — I verified the spec's own code did
not change during either run. What is affected is the ability of any future
auditor to check that the arms of a 17-hour bakeoff ran the same code. It is a
one-line class of fix, the correct value is already on disk, and it is B1 to
the builder. You are seeing it because it is the second time this exact hole
has been found in this exact spec, and the first repair is what failed.

**2. NO-DECISION — the honest reading of tonight's board.** Every instrument
this project owns is green (`decisions`, `champions`, `review-queue` all EXIT
0; the one `coverage` red is a known, dated, correctly-routed constitutional
citation). Twenty-five builder iterations in 24 hours, all `rc=0`, bought
**one** demonstrated capability. That is not a discipline failure — it is
§8's finding: **five specs across five families have now independently measured
that W0 is too shallow to falsify claims about living in it**, and the repairs
are all world redesigns, none of which a dispatch can buy. `D24` (your desk,
`decide_by` 2026-09-11) is the adjacent question and I have nothing to add to
the Review's recommendation of (iii).

**3. Still unanswered, re-stated not re-routed.** The `D8`/`D9` deadlock —
both defaults park the body question behind `LT.08`, which sits behind a chain
whose first link failed *because of the body*. The standing proposal to
register `W0.BAL` so the body gets a seat has been on this page since 08-31.
`balance` remains CLAIM-DEAD.
