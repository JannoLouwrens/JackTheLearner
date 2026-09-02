# OVERSIGHT — 60th audit, 2026-09-02 00:45 UTC (HEAD `5558e9e`, tree clean, 0 unpushed)

## VERDICT: DRIFTING — the backlog reader built so routed work could not go quiet is blind to **6 of the 26 rows in its own file**, and the 16.17 GPU-hours spent to unblock 35 specs bought a VOID that nobody is scheduled to repair

The counterweight first, because it is real and it is the larger part of this
audit. **Sections 1, 2, 6 and 7 were checked mechanically and section 2 is
completely clean**: across all of `experiments/` and `scripts/`, in the seven
days to now, **not one named numeric constant changed value in the loosening
direction** — the single constant edit in the window is `N_PROPERTIES 11 -> 12`
in `t0_31_review_queue_cannot_go_quiet.py`, a strengthening. Every one of the
93 PASS rows has an implementation on disk, a commit that still resolves in
git, and a declared control; the only two rows without control metrics
(`T0.01`, `T0.10`) declare `control="NONE, BY DECISION"` on their face.

And the thing this audit could most easily have been lazy about: **eleven armed
defaults fired on 2026-09-01**, and `SYSTEM.md` says in its own words that two
of the three safety clauses on a default "remain on the author's word" because
they are properties of the firing *commit*, which no instrument reads. So this
audit read them. **`GOAL.md` has not been touched by any commit since
2026-08-10**, and the constant scan above covers all eleven firing commits.
Both unenforced clauses hold, verified rather than believed.

The findings below are ranked by damage to the trustworthiness of the ledger
and its instruments.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | 0 commitments with NO spec; **4 CLAIM-DEAD** (smell, balance, shelter/building, thermal); 8 more with live claim specs and nothing passing. Standing red by design until the 09-06 Review window. |
| `decisions --check` | 0 | 3 armed (`D15` due 09-05, `D16` due 09-05, `D17` due 09-07). 0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE. Ratchet **0/10 undeclared**. |
| `champions --check` | 0 | 6 violations: 1 ARENA-UNREACHABLE (Fast/slow coupling, rooted at `LC.03`), 3 NO-ARENA (ASR, Speaker ID, Language grounding), 2 UNCONTESTED (Vision encoder, PLASTIC-ONLY). Ratchet ok. |
| `run review-queue` | 0 | Prints **16 OPEN, 2 HELD, 2 ACTED of 20 routed**. **The file contains 26 routed rows. See FINDING 1.** |
| `review_liveness` (schedule half) | 0 | Review kept its schedule; last DAILY 2026-09-01, log `review.log` 09-01 06:47. |

**On "arm at least one decision per audit":** vacuous today and stated rather
than faked. `decisions.py` reports **0 of 10 undeclared** — every open entry
already carries a class, a default and a `decide_by`. There is nothing to arm.
I did not invent an entry to satisfy the instruction.

---

## FINDING 1 (rank 1) — `run review-queue` cannot see 6 of the 26 routed rows in `docs/REVIEW_QUEUE.md`, and all six go STALE next week with nothing going red

**The evidence, counted twice.** `review_queue.py` parses `_ROUTED = re.compile(r"^ROUTED:\s*(.*)$")` — a declaration at column 0. The file has **20** such lines. It also has **six further routed sections**, each with its own `## ` header, each written in the pre-declaration prose idiom, and **each saying `Status: OPEN` or `ROUTED: OPEN` in its own words**:

| line | row | routed | own status text |
|---|---|---|---|
| 404 | `t310-anticorrelated-gates` | 2026-08-30 | routed by the spec's own pre-registered fork (ii); T3.10 PARKED |
| 456 | `SM.03`'s saturated held-out split — pick the repair arm | 2026-08-30 | **"Status: OPEN.** Gates provisional, `run()` still refuses" |
| 493 | should a PRESERVED failing implementation count as `audit_supersedes_fail`'s artifact | 2026-08-30 | **"Status: OPEN.** No gate was moved" |
| 527 | `sh02-null-saturation` | 2026-08-30 | header reads `## ROUTED: OPEN —` |
| 673 | `pl02-dependency-on-pl00-verdict-vs-table` | 2026-08-30 | header reads `## ROUTED: OPEN —` |
| 772 | `champions-language-grounding-arena` | 2026-08-31 | header reads `## ROUTED: OPEN —` |

Three of them literally begin `## ROUTED: OPEN` — the declaration is there, one
`## ` away from being read.

**So the true backlog is 22 OPEN of 26, not 16 of 20 — the printed count
understates the live desk by 27%.** None is old enough to be STALE today
(oldest 3 days against `MAX_OPEN_AGE_DAYS = 8`), which is exactly why this is
worth catching now rather than on 2026-09-07, when all six cross the line
**and no number will move**, because a row the parser never saw cannot age.

**Why this is rank 1 and not housekeeping.** This module's own docstring opens
*"A backlog nobody can count is indistinguishable from a backlog nobody has"*
and was written because `w0-too-shallow`'s dated promise passed in silence. The
same failure is now one layer down: the file holds rows, the reader counts a
subset, and the difference is invisible. The choice to require a declaration
rather than regex prose was **correct** — `champions.py` paid for the regex
version — but the two other instruments that made that same choice each ship an
`UNDECLARED` class for exactly this residue (`decisions.py` UNDECLARED;
`champions.py` prints *"UNDECLARED — no SEAT:/HELD:/ARENA: line … (0/0)"*).
`review_queue.py` alone has no such class, so an unmigrated row is not a
violation — it is simply not a row.

**Two of the six are load-bearing right now.** The `preserved failing
implementation` row (line 493) is the routed form of the question FINDING 5
below turns on, and it is invisible to the desk that must answer it. The
`champions-language-grounding-arena` row (line 772) is the builder **declining
an overseer order** (51st audit B2, `NO-ARENA` ×3) with a reasoned argument —
a legitimate move, and `champions --check` still reports those three NO-ARENA
violations today. That disagreement is supposed to reach the Review. It cannot.

---

## FINDING 2 (rank 2) — `D1.0` VOIDed, and the project's largest unblock now has no successor, no clock and no queue row

**What was bought.** `D1` fired as an armed default on 2026-09-01 and registered
`D1.0`, the four-arm control-path bakeoff, precisely so `T2.01` could be re-run
under a winner. It ran across three Kaggle kernels for **16.17 GPU-hours — 54%
of the entire 30 h weekly quota** — and returned **VOID**: `c_e2e` at 2.56σ
against a 3.0σ learning gate, so the rig correctly refused to arbitrate.

**What that leaves.** `run blocked`, live:

    T2.01 = FAIL  frees 35  (blocks 38)  — Locomotion beats a random policy

Unchanged. `85 of 217 specs unreachable`, exactly the ratcheted baseline. The
Control-architecture seat reads **VACANT**, arena `D1.0`, and `champions --check`
marks it **ok** — because `D1.0` exists and ran. Nothing asks whether a VOID
arena with no scheduled re-run is a live contest.

**And the two rows that were routed from it explicitly disclaim the repair.**
Both `d10-learning-gate-uses-two-different-denominators` and
`d10-learning-gate-sits-at-the-untrained-twin-level` (DUE 2026-09-06) end with
the same scope note: *"gate redesign for FUTURE runs only — the recorded VOID
stands per T2.02 precedent and nothing re-runs on this row's account."* That is
correct row-hygiene and it is **not** the missing thing. `SYSTEM.md`'s rule for
a VOID is *"fix the arm, do not decide"* — the arm is `c_e2e`, and I can find no
row, no `DUE:`, no `PROGRESS.md` item and no priority-block line that owns
fixing it or re-running the bakeoff. Grepping `docs/REVIEW_QUEUE.md` for
`T2.01` returns three hits, all incidental references inside other rows'
prose; `c_e2e` appears twice, both inside the gate-design row that disclaims
re-running.

The largest single unblock in the project spent 54% of a week's free GPU quota,
returned an honest VOID, and became nobody's work in the same motion. That is
not dishonesty — every number is correct and disclosed — but it is precisely the
`D1` disease in its non-escalated form: a thing everyone can see and no
instrument owns.

---

## FINDING 3 (rank 3) — the board was declared "genuinely empty until the 09-06 Review" while the builder's own instrument named four implementable specs with every dependency PASS

The 00:07 iteration closed with: *"The board is now genuinely empty until the
09-06 Review … the next iteration should verify liveness and stop honestly
rather than manufacture a unit."* Read against `coverage.py`'s queue block in
the **same** output:

    cpu<10min   0   EMPTY   <- fillable today: LG.10, ME.11
    cpu<2h      0   EMPTY   <- fillable today: LG.02, T3.09

Checked live against `BY_ID` and the ledger:

| spec | tier | budget | impl | deps |
|---|---|---|---|---|
| `LG.10` Jack chooses what to say; the LLM only chooses how | 4 | cpu<10min | none | `LG.00` **PASS** |
| `ME.11` Finds the memory from a paraphrase, still never invents one | 2 | cpu<10min | none | `ME.1`, `ME.11.0` **PASS** |
| `LG.02` Trust is earned by track record — the liar loses him | 5 | cpu<2h | none | `ME.9` **PASS** |
| `T3.09` The creative loop earns its existence | 3 | cpu<2h | none | (none) |

All four are unimplemented, unblocked, CPU-only, and free. **"Empty board" and
"no work" are not the same claim**, and the distinction matters because the
posture announced covers four days at 25 iterations a day — roughly **100
iterations**. The standing instruction not to manufacture busywork is right;
`coverage.py` naming four fillable specs is not busywork, it is the instrument
doing its job and being read past.

Two of the four are not filler. `LG.02` — *"his diary records whose advice
proved true, so trust in a person can be earned and checked"* — and `LG.10` are
GOAL.md commitments in the language family, which currently has 2 passing claims
of 9 specs.

---

## FINDING 4 (rank 4) — `LC.07` still declares `_PILOT_OWED` after its pilot was harvested and its branch forbade freezing, so `coverage.py` advertises already-spent work as the cheapest repair available

`coverage.py`, live, at the top of the GPU queue:

    gpu<8h  2  D1.0, T2.02  <- PILOT ALREADY RAN, HARVEST IT (cheapest repair of all): LC.07 -> /data/lc07_pilot.json
    1 spec(s) are PILOT-HARVESTABLE … The run is already spent; the next unit is
    to read it and either freeze the gates or declare `_PILOT_BLOCKED`

That harvest happened on 2026-09-01 at 22:0x for 0.44 GPU-hours. The spec's own
docstring now carries `================= PILOT RECORD (BRANCH B FIRED — NOTHING
FROZE) =============`: the cheapest run class projects **14.49 h** and the arm
**40.86 h** against rule A's **8.5 h** kernel ceiling, so `_GATES_FROZEN` stays
`False` and `run()` keeps refusing. But `_PILOT_OWED` was never retired, so
`coverage.py` — whose own docstring records `_PILOT_OWED` *"went on asserting
no pilot has been run"* as a past scar — is reproducing that scar and pointing
the next iteration at a unit that is finished.

`_PILOT_BLOCKED` is the state that fits, by its own definition (*"a run has
MEASURED that the pilot's own precondition fails … the repair is a redesign"*),
and `DP.04`, `SH.02`, `SM.03` and `T2.11` all declare it for exactly this shape.

---

## FINDING 5 (for the owner, bearing on `D16`, due 2026-09-05) — a second live `T0.27` violation appeared **one day after** option (b) shipped, and its failing bytes are recoverable

`D16` was filed 2026-08-29 on one pair (`T0.17`), reading *"4 checked pairs, 26
unauditable, 1 violation"*. Its armed default is **(b) alone — the warning
stands, `T0.27` stays RED**. Live now:

    audit_supersedes_fail: 2 violations, 7 checked, 24 unauditable
      LG.00   VOID  8faff43+dirty  2026-08-30T18:47:59
      T0.17   FAIL  d84101e+dirty  2026-08-29T13:14:23

`LG.00`'s adverse verdict was recorded **2026-08-30**, after the dirty-run
warning was live. The warning did not prevent the recurrence. That is one
counter-example, not a refutation — but the owner is about to rule on 09-05 on
the premise that (b) reduces the rate, and this is the only measurement bearing
on it.

**Second fact, in the other direction, and the more interesting one.**
`LG.00`'s failing implementation is **not lost**: it is preserved and verified
at `refs/jack/failimpl/LG.00/2026-08-30T18-47-59` (blob `d39a0ef`), written by
`preserve_impl_bytes`, which re-derives `impl_sha_of` from the stored bytes and
refuses to write the ref unless it equals the sha the row names. Its firing
commit `6c008d9` publishes the exact per-seed table (26, 22, 21), states that
`RETAIN_MIN` stayed at LG.01's 20 and `SIGMA_MIN` at 3.0, and adds a new VOID
gate. So for that pair the `git diff` the rule demands **is** possible, and the
auditor's stated reason — *"that implementation was never committed"* — is
false as written. `T0.17`'s bytes really are gone; `LG.00`'s are not, and the
instrument cannot tell the two apart.

**I am not recommending the change.** Whether a preserved manifest is an equal
artifact is CONDUCT, it is the owner's, and it is already routed — in
`docs/REVIEW_QUEUE.md` line 493, which is one of the six rows FINDING 1 shows
the desk cannot see. Both halves of this finding are here so the 09-05 ruling is
made on the measurement rather than on the 08-29 snapshot.

---

## THE AUDIT, SECTION BY SECTION

### 1. Integrity of the ledger — NO FINDINGS
93 PASS / 18 FAIL / 10 VOID over 121 rows. Checked mechanically over all 93
PASS rows: **0** with a missing implementation file, **0** whose recorded
`commit` no longer resolves (`git cat-file -e`), **0** whose spec declares no
control. Two carry empty `control_metrics` — `T0.01` and `T0.10` — and both
declare `control="NONE, BY DECISION (52nd audit B5)"`, which is a recorded
ruling, not an absence.

Staleness, as `run status` reports it and I confirmed: 3 CHANGED (`T0.27`,
`UB.10`, `D1.0` — all three are rows whose test was deliberately edited after
an adverse verdict and are accounted for), 1 stale-by-content (`T2.02`,
historical, standing do-not-re-run), 15 pre-`impl_sha` of which **14 verified
byte-identical by git** and 0 unanswerable. **54 of 93 PASS rows predate
`spec_sha`** and the record cannot say whether their claim text moved — printed
honestly by the tool, carried here as a standing limitation rather than a new
finding.

### 2. Thresholds and controls over time — NO FINDINGS, verified mechanically
Diffed the last 7 days of `registry.py`, `registry_expansion.py` and
`experiments/tests/`, and separately every commit on 2026-09-01 across all of
`experiments/` and `scripts/`, extracting every `NAME = value` whose value
changed. **Result: one change in the whole window** —
`t0_31_review_queue_cannot_go_quiet.py: N_PROPERTIES 11 -> 12`, a property
added. No `control=` string was weakened (`T2.10`'s is a reflow inside a
strengthen-only redesign that ADDED a conjunct and a leaky-cue aliveness floor).
No `depends_on` was loosened; the one edit — `BA.02` gaining `LT.08` under D8 —
adds a blocker. No `seeds=` was reduced. No assertion was deleted.

This section is where silent loosening would live, and this week there is none.

### 3. Drift from the goal — NO DRIFT; the converse is the problem
Every unit in the last 24 h traces to a GOAL.md sentence: `D1.0` and its harvest
→ *one interconnected brain* / the control path; `UB.10`'s unpark, grid pilot,
registered run and VOID-routing → *all senses in unison*; `LC.07`'s pilot → the
owner's scale-transfer guard on the learning core; `GEN.02/03/06/09` →
*the jungle is the foundation, not the destination*; `T1.09`/`T1.10` re-aim and
re-buy → harness honesty; the orphaned-dispatch detector and the 59th audit's
B1–B7 → *protects the honesty of watching what happens*. Nothing served none.

The converse, from `coverage.py`, is the standing wound and it did not move:

- **4 CLAIM-DEAD** — smell, balance, shelter/building, thermal(kills). Every
  claim spec parked or foreclosed; the passing specs in those families are
  fixtures and sensors, which are support, not claims.
- **8 with live claims and nothing passing** — touch, tool use, proprioception,
  death & retry, plasticity, sleep, hunger/thirst, fast/slow.
- **one brain / unison: 23 specs, 1 passing.** **curiosity: 12 specs, 2
  passing** — and `LT.01`, the instrument the whole `LT.03`/`LT.04` ladder-test
  family sits behind, FAILED on 08-31 and now blocks 9.
- `CITED-BUT-UNRUNNABLE`: `DP.02`, `DP.03`, `LC.04` — three ids GOAL.md cites in
  the present tense that resolve to welded specs. Carried from the 59th audit,
  routed as `reparenting-the-welded-fifteen` (DUE 09-06).

### 4. Is the builder alive and productive? — ALIVE, PRODUCTIVE ON THE MACHINE, FLAT ON THE LADDER
**25 iterations in the 24 h to 00:15, 25 of 25 ended `rc=0`, zero aborts, zero
credit exhaustion, zero PACING skips** (fresh week: `week:all models` 5% at the
last slot, and the gate acted on is correctly the all-models line, not the
per-model one). Load never exceeded 0.31. No orphaned processes; `run status`'s
new ORPHANED DISPATCHES block is silent.

**PASS delta over 24 h: 93 → 93, zero.** Four consecutive declared hold slots
(15:07–18:07) behind the `D1.0` GPU lock, each honestly reported. Over 7 days:
**PASS 84 → 93 (+9) while the registry went 187 → 217 specs (+30)** — the
denominator is growing **3.3×** faster than the numerator. That is not
automatically bad (registering `LT.01–LT.09` and the `GEN` family discharged
real `champions`/citation violations) but it is the number to watch, and
FINDING 3 says the flat 24 h was not forced.

### 5. Compute honesty — ACCOUNTED IN FULL; 17.31 of 18.93 hours bought two VOIDs
Kaggle week **W35 (Sun 2026-08-30 → Sat 2026-09-05, `%U` Sunday-start, matching
Kaggle's reset): 18.9304 of 30 h spent, ~11.07 h remaining, resets Sunday
2026-09-06.** `overruns` is empty. Every W35 job carries a `charged_jobs` row;
no unattributable hours this week (the only `opening_balances` gap is W32's
6.3849 h, frozen and disclosed since the 16th audit).

Attribution, hour by hour:

| spend | what it bought |
|---|---|
| 16.17 h (3 kernels) | `D1.0` → **VOID** ledger row |
| 0.70 h | `UB.10` grid pilot + registered run → **VOID** ledger row |
| 0.44 h | `LC.07` throughput pilot → no ledger row; branch B fired (correct) |
| 0.11 h | `T1.09` + `T1.10` re-buys → **2 PASS** rows on the P100 |
| 0.50 h | `D1.0` envelope pilot → three-kernel escalation, fired loudly |

**No hours are missing and nothing is hidden.** But 17.31 of 18.93 h produced
two VOIDs and no unblock, and the third dead-watcher orphan of the project
happened inside this window (`T1.09`, recovered via `JACK_REUSE_KERNEL`, cost
0.05 h re-metered, and repaid with a harvest-side detector). The honest reading:
the accounting is trustworthy, the *yield* is the concern, and FINDING 2 is
where that concern lands.

### 6. Stuck decisions — NO VIOLATION; two notes
Nothing MEANS-ESCALATED, nothing UNDECLARED, nothing OVERDUE. Eleven defaults
fired on 09-01 and all eleven were recorded in `DECISIONS_RESOLVED.md` with
losers attached; none was acted on without a record. Two notes:

- **`D15` (due 09-05) has partially overtaken itself.** Its default is "(c) AND
  (d)". `D13` resolved on 09-01 as the change-gated no-op and it is **live** in
  `scripts/overseer.sh:50–88`; the `usage_ledger.jsonl` half of `D15`'s (d) does
  **not** exist (`/data/jack-logs/usage_ledger.jsonl` absent). Also worth the
  owner knowing before ruling: the harm `D15` was armed against — 84 consecutive
  PACING skips — is **not occurring today** (fresh week, 5%, 25/25 iterations
  ran); it is a late-week condition.
- **`D17` (due 09-07)** — PLASTIC-ONLY's re-open trigger. Its default keeps the
  decree verbatim and keeps `PL.02` registered, so nothing goes claim-dead. No
  concern.

### 7. Bakeoff hygiene — ONE STRUCTURAL FINDING, disclosed but now un-dischargeable
`DECISIONS_RESOLVED.md` records no winner chosen inside a noise margin and no
decision made without a learning gate. The one shape this section exists to
catch **is** present and it is on its face rather than hidden:

**`D10` seated `wm-latent` on the Learning core seat marked BY VERDICT, from a
run whose ledger row is VOID** (`LC.03` v2, *"fewer than two learners (1
cleared)"*). The firing was scrupulous about it — "single-arm caveat on its
face", losers recorded, three pre-registered re-open triggers, and adoption
explicitly gated behind the owner's ~10× scale-transfer guard registered in the
same commit as `LC.07`. That is the honest way to do an uncomfortable thing.

**What has changed since, and it is not on any page yet:** `LC.07`'s pilot
harvested on 09-01 measured that the guard **cannot physically run in the only
free venue** — cheapest class 14.49 h, the arm 40.86 h, whole plan ~526 wall
hours, against an 8.5 h kernel ceiling. So the escape hatch that keeps a
BY VERDICT-off-a-VOID seating honest is currently unreachable, and
`champions --check` reports the Learning core seat **ok** because `LC.04`–`LC.07`
are `NOT_RUN` rather than welded. The `lc07-checkpoint-branch` row (DUE 09-06)
prices three options and correctly refuses to pick one. That row is the right
place; it just needs the Review to know that the *seat's* honesty, not only
`LC.07`'s runnability, is what is waiting on it.

### 8. The honest summary — are we closer to a curious humanoid, or to a longer list of ticks?
**Today: neither. We are closer to a more honest machine, and no closer to
Jack.**

The 24 h under audit added **zero** demonstrated capabilities. What it added
was: an orphaned-dispatch detector that reads the half of the receipt log
nothing read, a lock file that now means something, a `DISPOSITIONED` status
that ages, four `GEN` specs that turn constitutional citations into resolvable
ids, and two VOIDs that were reported truthfully instead of massaged. By
`SYSTEM.md`'s standard — *"any session that makes the machine better at catching
its own errors has done the whole job even if no spec passed"* — that is a
defensible day, and the honesty is genuine: `D1.0` and `UB.10` both returned
verdicts their authors did not want, with green rigs, and both were routed
rather than re-rolled.

But the ladder-and-apple standard is the one that matters, and against it: **the
ladder test itself (`LT.01`) is FAILING and blocks 9**; curiosity has 2 passing
claims of 12 specs; the unified brain has 1 of 23; four of the owner's own
sensory commitments are CLAIM-DEAD; `T2.01` — *he can move* — has been FAIL for
weeks and blocks 35. The seven days bought +9 PASS against +30 registered specs.
The machine is now very good at telling us precisely which parts of Jack do not
exist. That was the point of building it, and it has stopped being enough.

The specific thing that would make tomorrow different is not a new organ. It is
FINDING 2 getting an owner and a date, and FINDING 3's four CPU specs getting
implemented instead of the loop holding station until Sunday.

---

## FOR THE BUILDER — ordered

1. **Give `review_queue.py` an `UNDECLARED-ROW` violation class, and migrate the
   six rows.** A `## ` section in `docs/REVIEW_QUEUE.md` that contains no
   `^ROUTED:` line is currently not a row at all. Make it a counted violation in
   the same idiom `decisions.py` and `champions.py` already use for their
   undeclared residue, then add the missing `ROUTED:` lines (and a `DUE:` where
   the row's own prose implies one) for the six at lines 404, 456, 493, 527, 673
   and 772. **Do the class first and watch it fire on six**, then migrate — a
   repair that lowers its own number without ever having gone red is the exact
   pattern `T0.31` P4/P5/P6 exist to refuse. Keep the ratchet at 6 in the
   fixture. The three headers that already read `## ROUTED: OPEN —` are the
   evidence that a declaration inside a heading will keep being written; the
   parser should count them, not learn to read them.

2. **Route `D1.0`'s successor with a clock.** `D1.0` VOIDed on one arm
   (`c_e2e`, 2.56σ) and `T2.01` still blocks 35. The two `d10-*` rows correctly
   scope themselves to future gate design; neither owns the repair. Open one row
   naming the actual unit — repair `c_e2e` and re-run, or declare the bakeoff
   VOID-FORECLOSED with the arithmetic — with a `DUE:` in the 09-06 window.
   Right now the largest unblock in the project is owned by nobody, and the
   16.17 GPU-hours already spent are the argument for giving it an owner, not
   against it.

3. **Flip `LC.07` from `_PILOT_OWED` to `_PILOT_BLOCKED`** (or whatever
   declaration honestly names "pilot ran, branch B fired, the repair is a
   redesign"), quoting the branch-B numbers already in the docstring:
   14.49 h cheapest class / 40.86 h arm against the 8.5 h ceiling.
   `coverage.py` is currently advertising a finished 0.44 GPU-hour harvest as
   *"the cheapest repair of all"* at the top of the `gpu<8h` class, and the next
   iteration that trusts it will redo done work.

4. **Do not hold station until 09-06.** `coverage.py` names four specs
   implementable today with every dependency PASS and no GPU:
   `LG.10` and `ME.11` (cpu<10min), `LG.02` and `T3.09` (cpu<2h). `LG.02`
   — *trust is earned by track record* — and `LG.10` are GOAL.md commitments in
   a family with 2 passing claims of 9. "The board is empty" and "there is no
   work" are different claims, and the instrument printed the difference in the
   same output the empty-board conclusion was drawn from.

5. **(Small, and only after 1.)** When you migrate row 493, note on it that its
   question now has a second data point: `LG.00`'s failing bytes ARE preserved
   and verified at `refs/jack/failimpl/LG.00/2026-08-30T18-47-59`, so of the two
   live `T0.27` violations one is recoverable and one is not, and
   `audit_supersedes_fail` reports both with the same sentence. Do not change
   the gate — that is `D16`, and it is the owner's.

---

## FOR THE OWNER

**1. `D16` fires 2026-09-05, and the measurement it rests on has moved. Read
FINDING 5 before it does.** You armed *(b) alone — the warning stands, `T0.27`
stays RED* on the reading that the dirty-tree warning would reduce the rate. A
**second** violation was created on 2026-08-30, the day after the warning
shipped: live audit now reads **2 violations / 7 checked / 24 unauditable**,
against **1 / 4 / 26** when the entry was filed. Separately, the newer of the
two violations (`LG.00`) has its failing implementation **preserved and
cryptographically verified in the repo**, and its firing commit publishes the
per-seed table and states that no threshold moved — so for that pair the audit
trail the rule demands exists, while the instrument reports it with the words
*"that implementation was never committed."* The default remains honest and
costs only a visible red row; it is simply now resting on a premise with one
counter-example. Appended to `docs/DECISIONS_NEEDED.md` with the evidence.

**2. `D15` fires 2026-09-05. Two facts that were not true when it was armed.**
`D13` resolved on 09-01 and its change-gated no-op is **live** in
`scripts/overseer.sh`, which is part of what `D15`'s (c) asks for; the
`usage_ledger.jsonl` attribution half of (d) does not exist yet. And the harm
`D15` was armed against — 84 consecutive PACING skips, 35 of the 65-point band
consumed by the oversight organs — **is not occurring in the current week**
(builder at 25/25 iterations, `week:all models` 5%). It is a late-week
condition, so the ruling matters, but the urgency reads differently than the
08-29 measurement implies.

**3. A seat is held BY VERDICT off a VOID, and its escape hatch has since been
measured impossible.** `wm-latent` holds the Learning core seat by the `D10`
default that fired on 09-01 — honestly disclosed as single-arm, with adoption
gated behind your ~10× scale-transfer guard. On 09-01 that guard (`LC.07`) was
measured as physically unrunnable in the only free venue: 40.86 h for one arm
against an 8.5 h Kaggle kernel ceiling, ~526 wall-hours for the plan. The
builder correctly refused to pick a repair and routed
`lc07-checkpoint-branch` (DUE 09-06) with three priced options, one of which —
*"a Review/owner re-read of what '~10x' means"* — is explicitly yours. This is
not urgent this week; it is the thing that decides whether the learning-core
seating stays honest.

**4. Nothing was weakened, and I checked rather than assumed.** No threshold
moved in seven days. `GOAL.md` is untouched since 2026-08-10. All eleven
defaults that fired on 09-01 respect both of the safety clauses that
`SYSTEM.md` says no instrument enforces. The ledger's 93 PASS rows all have
implementations, resolvable commits and declared controls. If you read one line
of this report as reassurance, that is the one — and FINDING 1 is the one to
read as a warning, because it is an oversight instrument under-reporting its own
subject matter by 27%, which is how the last three of these were found.
