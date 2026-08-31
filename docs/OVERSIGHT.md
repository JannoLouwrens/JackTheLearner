# OVERSIGHT — 55th audit, 2026-08-31 18:45 UTC (HEAD `459eeb1`, tree clean)

## VERDICT: DRIFTING — integrity is clean on every hard check, and the ladder spent 40% of a day's iterations on its own instruments while the thesis sits at 2 passing specs of 44

Sections 1, 2, 6 and 7 have **no findings**, checked mechanically. I say this
first and plainly because it is true, it is the most valuable result in this
report, and it is what makes the rest worth reading. Specifically:

- All **94 PASS** rows resolve to a live commit; **0** name a commit git has lost.
- **94 of 94** declare a `control`. **92** carry non-empty `control_metrics`.
  The 2 that do not — `T0.01`, `T0.10` — declare `NONE, BY DECISION (52nd audit
  B5)` with the reasoning on the spec. That is a documented hole, not a hidden one.
- **No silent loosening in 7 days.** Every constant that moved, moved in the
  tightening direction, each with a measurement in its commit message:
  `N_DECISIONS` 3200→4800 (W0.DIAG v2 envelope ×1.5, floors unchanged),
  `N_EVAL` 48→120 (BA.03, derived from the pilot's own sigma),
  `HEADROOM_MIN_MULT` 2.0 (new, strengthen-only),
  `CHANCE_BAND_HI` 0.25→**0.0** (LG.01 — the null must now be outright wrong,
  not merely at chance), `N_PROPERTIES` 8→9→11. `LG.00`'s `_check` moved from
  mean±std to per-seed `min`/`max` on all six conjuncts — strictly stricter.
  `BA.03`'s conjunct 7 (`gain_os <= NOSURF_GAIN_MAX`) is intact. No `_check`
  gained an `or`, no seed count fell, no control was deleted.
- **No deadline was ever moved.** Twenty-one days of `docs/DECISIONS_NEEDED.md`
  history contain **zero** `-  decide_by` lines. Every one of the eleven armed
  defaults still carries the date it was armed with.

The defect is not in the ledger. It is in where the iterations go.

**The four mandatory ratchets.**

| tool | rc | reading |
|---|---|---|
| `coverage` | **2** | **RED**, honestly — 4 empty classes with no path in. Constitutional check green: **0 commitments with no declared spec** |
| `decisions --check` | 0 | 0 UNDECLARED, 0 MEANS-ESCALATED, 0 OVERDUE. **Eleven defaults fire tomorrow**, `D1` among them (38 specs) |
| `champions --check` | 0 | `ratchet ok`, 8 violations all pre-existing and declared |
| `run review-queue` | 0 | 0 violations — **and the backlog grew from 7 rows to 13 while reporting zero** (§4b) |

Every ratchet is green or honestly red. The loss is again happening in the space
*between* the instruments — and this time one of the instruments is me.

---

## 1. Integrity of the ledger — NO FINDINGS

See above. Checked mechanically over all 94 PASS rows of 211 registered specs:
implementation resolves, commit lives, control declared, control exercised.

## 2. Thresholds and controls over 7 days — NO FINDINGS

See above. 41 commits touched `registry*.py` / `tests/`; every existing constant
that moved, tightened.

## 3. THE FINDING — the audit organ is the largest single consumer of builder iterations, and nothing counts its cost

**25 iterations in 24 h** (2026-08-30 18:07 → 2026-08-31 18:26), **all `rc=0`**,
zero deaths, zero credit exhaustion, no leftover processes. The loop is healthy.

What it built:

| work | iterations | outcome |
|---|---|---|
| 54th-audit B1–B6 (my own B-items) | 6 | `3688b9e` `404e25a` `aabced4` `78aad78` `abb3d70` `4eda7cb` — all instrument/doc code |
| certificate re-buys forced by those same edits (`IMPL_DEPS` decay) | ~5 | `97e7629` `da84cb9` `9d10435` `97c1630` `dad5f27` — T0.17, T0.21 ×2, T0.29 re-bought |
| spec implementations | 3 | ME.11.B, ME.11.C, ME.11.D — **all three settled FAIL** |
| W0.DIAG (53rd audit / Review priority 1) | 1 | PASS |

**Roughly 11 of 25 iterations — 44% of a full day — went to overseer-generated
instrument work and the certificate re-buys it mechanically causes.** Three went
to specs about Jack.

The mechanism is not misconduct and the builder is not idling. It is arithmetic:
every edit to `coverage.py` / `champions.py` / `protocol.py` / `run.py` decays
the `IMPL_DEPS` of the T0.* certificates that gate those files, and each decayed
certificate must be re-bought in a later iteration. **I generate the edits and
the ladder pays twice.** The 54th audit issued six B-items in one report; that is
what a 44% day looks like.

**No instrument in this repo measures overseer-induced load.** `coverage`,
`decisions`, `champions` and `review-queue` all count the system's debts to
itself. None counts the debt this organ creates. That asymmetry is why it took
55 audits to notice — and it is exactly the shape of hole this role exists to
find, with the uncomfortable feature that the hole is mine.

**Consequence, taken here rather than recommended:** this audit issues **three**
B-items, not six, and none of them edits an instrument.

### 3b. The PASS delta, honestly

**91 → 94 PASS (+3) while the denominator went 200 → 211 (+11).** The
demonstrated fraction *fell*, 45.5% → 44.5%. Of the +3, two are T0.* harness
certificates (`T0.31` the backlog reader, plus re-buys) and one is `W0.DIAG`, a
diagnostic about the world rather than a capability of Jack's.

Composition of the 94: **30 are T0.\* (32%) — the harness measuring itself.**
By tier: 32 Tier-0, 13 Tier-1, 43 Tier-2, **1 Tier-3, 2 Tier-4, 2 Tier-5**, 1 Tier-6.

## 4. Drift from the goal

### 4a. What GOAL.md still has nothing behind it

`coverage` reports 0 uncovered commitments — the constitutional floor holds. But
the harder converse question:

**Tier 5, "THE CLAIMS — the thesis itself": 44 specs, 2 PASS.** (`TA.02`
one-trial aversion, `VO.02` two Jacks invent a signal.) 3 VOID, 1 FAIL,
**38 NOT_RUN.**

The north star specifically:
- **The ladder-and-apple standard has zero passing specs.** `LT.01`–`LT.09` were
  registered *today* (`3688b9e`) and all nine are NOT_RUN. `LT.01` — the one that
  makes the rest measurable — is the third-largest blocker in the project
  (frees 7) and is `NOT_RUN` with a `cpu<2h` cost.
- **The entire curiosity family `CU.1`–`CU.7` is NOT_RUN**, all seven blocked
  behind `T2.01`.
- **`T5.06` "Unprompted exploration is real"** is welded shut behind `T3.06`'s
  VOID-FORECLOSED declaration.

**80 of 211 specs (38%) are unreachable.** One FAIL dominates: **`T2.01`
"Locomotion beats a random policy" blocks 37 and frees 35 alone** — every CU
spec, `T5.01`–`T5.05`, `T5.07`, `T6.01`–`T6.05`, `UB.1`–`UB.8`. It last ran
**2026-08-12, nineteen days ago**, and its metrics record
`all_seeds_beat_random: 1.0` — it *did* beat random on every seed, at 4.06σ
against a bar of 5. It is FAIL on margin, not on direction. It is blocked behind
`D1`, whose default fires tomorrow. That is the correct sequencing and I flag it
only so the size of tomorrow is not underestimated.

### 4b. The Review queue is growing 13× faster than it is being cleared

`run review-queue` returns **0 violations**, correctly. It is also true that:

- **13 rows routed in 11 days** — 1.18/day arriving.
- **1 ACTED** in that entire period — 0.09/day disposed.
- The queue went from 7 rows (when the 54th audit ran, ~6 h ago) to **13**.

The reader is working exactly as built and reports green, because no row has yet
aged past a clock. But the desk my own B4 built to make the backlog countable
now counts a backlog compounding at roughly **13:1**. "A desk can open every
morning and dispose of nothing" is in my own brief; this is the measured version.

### 4c. Four dated promises are staked on one untested run of a mode that is 0-for-12

Every `DUE: 2026-09-06` row in `REVIEW_QUEUE.md` — **all four** — says the work is
*"owed by the Review's Sunday FULL run"*:

    w0-too-shallow                              DUE 2026-09-06
    ba03-null-saturates-the-horizon             DUE 2026-09-06
    t306-matched-magnitude-noise-buys-coverage  DUE 2026-09-06
    reparenting-the-welded-fifteen              DUE 2026-09-06

`docs/PROGRESS_LOG.md` contains **12 DAILY rows and 0 FULL rows.** `review_liveness`
says it in one line:

    REVIEW LIVENESS FAILED — no FULL row has EVER been written to
    docs/PROGRESS_LOG.md — that mode has never completed

**The FULL mode has never once completed in this project's history** — four
Sundays, four deaths. To be fair to the builder: the cause was diagnosed and
repaired on 2026-08-30 (`review.sh` now derives `MAXTURNS = MINUTES × 3`, so FULL
gets 120 turns instead of the hard-coded 60 that killed it at 11 minutes of a
40-minute budget). That repair is real and correctly reasoned.

**It has never been executed.** 2026-09-06 will be its first run — and four dated
promises, two HELD rows behind `w0-too-shallow`, and the world redesign that five
specs need are all staked on it succeeding first time. If it dies a fifth time,
`run review-queue` reports **4 OVERDUE simultaneously**, its strongest violation
class, on a day when nothing is scheduled to repair it.

## 5. Compute honesty — a fourth consecutive week of free quota will expire

    2026-W32:  16.61 h / 17 jobs  (1.18 h on failed jobs)
    2026-W33:   7.89 h / 22 jobs  (0.26 h on failed jobs)
    2026-W34:   1.62 h /  4 jobs
    2026-W35:   1.28 h /  3 jobs   <- current, ~28.7 h of 30 unspent

No GPU hours were spent without a ledger entry — spend is honest. The waste is
the opposite: **~28.7 h of free Kaggle quota will expire at 04:59Z Sunday**,
because `coverage` reports both GPU classes **NOT FILLABLE** — `gpu<20min` and
`gpu<2h` are pilot-blocked on measured evidence (`DP.04`, `SM.03`, `T2.11`) whose
repair is a redesign, not a dispatch. Usage has fallen 16.6 → 7.9 → 1.6 → 1.3 h
across four weeks. This is D15's subject and is correctly on the owner's desk;
I record the fourth week rather than re-escalate.

## 6. Stuck decisions — NO FINDINGS

**Eleven armed defaults reach `decide_by: 2026-08-31` today.** None is OVERDUE:
the tool reads the date inclusively, so today is the owner's last day and the
defaults fire 2026-09-01. The builder has correctly and explicitly declined to
fire them early in each of the last five iterations, naming the reason each time.
Nothing was silently extended (verified above: zero `decide_by` edits in 21 days).
0 MEANS-ESCALATED, 0 UNDECLARED. Every entry names its executor and firing
artifact (`03f31cf`). This section is clean and the sequencing is right.

## 7. Bakeoff hygiene — NO FINDINGS

No decision recorded without a learning gate; no VOID treated as a verdict — the
`D2` resolution exists precisely to forbid that and is enforced in `T0.08`
property 6. No winner chosen inside a noise margin.

## 8. A separate finding — the ME.11 bakeoff's arms keep failing the same rig gate, and nothing counts the invariant

Three arms settled FAIL today. Each was journalled as informative about that arm.
Across the family the reading is different:

| arm | config | recall@1 | unthresholded ceiling | conformal feasibility |
|---|---|---|---|---|
| B | bm25s + Snowball | 0.0000 | — | n/a |
| C | potion-base-8M | 0.0437 | 0.123 | `tau_fpr 0.365 > tau_cov 0.184`, **3/3 seeds** |
| C var | potion-base-2M | 0.031 | — | INFEASIBLE |
| C var | mrl-en-v1@256d | 0.015 | — | INFEASIBLE |
| D | all-MiniLM-L6-v2 | 0.0667 | 0.250 | `tau_fpr 0.388 > tau_cov 0.227`, **3/3 seeds** |
| D var | bge-small | 0.067 | — | INFEASIBLE |

**Five distinct encoder configurations, spanning static and contextual, all hit
the same pre-registered INFEASIBLE branch on all three seeds.** When every arm of
a bakeoff fails the *same rig gate in the same direction*, the invariant is
evidence about the rig at least as much as about the arms.

The arithmetic that is not written down anywhere: **`ME.11`'s parent hypothesis
requires paraphrase recall ≥ 0.80. The best unthresholded ceiling any arm has
measured is 0.250** — that is with abstention disabled entirely, i.e. the
credulity-free maximum. The target is 3.2× above the measured ceiling of the best
arm tried.

The two remaining arms inherit this:
- **`ME.11.E`** fuses Arm B (0.0000) and the best dense arm (0.0667) against an
  0.80 bar. The builder's own journal already records the premise as dead
  (*"fusing two ~0.05 parents"*).
- **`ME.11.F`**'s pilot number (cascade p@1 0.875, *"the only configuration that
  cleared ME.11's 0.80 hypothesis"*) comes from the same pilot family in which
  ME.11.D falsified **three** numbers (485 docs/s not 93; reindex 206 s not 18 min;
  int8 faster, not slower) and ME.11.C killed the *"MiniLM ties Arm C at 0.625"*
  premise outright.

This is the repo's own filed lesson — *"a gate can be too STRONG to be met, and
nothing in this repo looks for that"* (`f66a5be`) — recurring at family scale.
**There is no `ME.11` row in `REVIEW_QUEUE.md`.** Nothing has routed it.

Related and small: `CHAMPIONS.md`'s Episodic-retrieval seat still reads
*"potion-8M favourite, cascade the risk"* after potion-8M measured 0.0437 and
INFEASIBLE this afternoon. The prediction column is checked against the ledger by
no instrument.

---

## FOR THE BUILDER

Three items, deliberately — see §3. **None of these edits an instrument**, and
none should trigger an `IMPL_DEPS` re-buy. If any does, do the cheaper thing and
say so.

**B1 — rehearse the Sunday FULL run before Sunday.** (§4c) The `MAXTURNS`
repair in `review.sh` is sound and untested, and four dated promises plus two
HELD rows are staked on its first execution. Run `review.sh` off-schedule with
`MODE=FULL` forced (or `JACK_REVIEW_MODEL` set as you prefer), on any day before
09-06, purely to learn whether 120 turns and 40 minutes are enough to write a
`FULL` row to `PROGRESS_LOG.md`. Report the turn count and wall time it actually
used. If the rehearsal also dies, the honest repair is to **re-arm the four
09-06 rows with new dates and reasons** — that is a permitted disposition and
deleting a clock is not. Do not let four promises come due on an untested run.

**B2 — route the ME.11 family signature to the Review, and measure the one
number that settles Arm F.** (§8) Open a `REVIEW_QUEUE.md` row —
`me11-every-arm-hits-the-same-infeasible-branch` — carrying the table in §8: five
configurations, all `tau_fpr > tau_cov` on 3/3 seeds, best unthresholded ceiling
0.250 against a parent hypothesis of 0.80. Before implementing **either** E or F,
measure and record **Arm C's `recall@50` on the certified stem-disjoint
fixture** — your own journal names it as the number that decides whether F's
premise survives, and it is cheap. If recall@50 is low, F cannot rerank what was
never retrieved, and both remaining arms are known-outcome runs. State plainly in
the row whether E is arithmetically reachable at all.

**B3 — correct the Episodic-retrieval seat's prediction cell.** (§8) One line in
`CHAMPIONS.md`: *"potion-8M favourite"* is falsified by `e3824bf`. Replace it
with what the arena has actually measured. This is a doc edit, not an instrument
change.

## FOR THE OWNER

**1. Eleven pre-registered defaults fire tomorrow, 2026-09-01.** Today was the
last day named on each of them when they were armed, and I verified mechanically
that **not one deadline has been moved or removed in twenty-one days** — the
clock you were given is the clock that is running. The largest is **`D1`, which
costs 38 specs**: the whole curiosity family, the unified-brain family, and six
of seven Tier-5 claims. Its default keeps the PLASTIC-ONLY decree verbatim and
sends the remaining arms to a bakeoff. If you want a different outcome on any of
the eleven, tomorrow is past the date. Each default is reversible and says how
in its own entry.

**2. The thesis is at 2 passing specs of 44.** Tier 5 — *"the claims"* — holds
`TA.02` and `VO.02`. The ladder-and-apple standard that GOAL.md calls the north
star has **zero** passing specs; its nine `LT.*` specs were registered today and
none has run. The curiosity family is seven-for-seven NOT_RUN behind `T2.01`,
which last ran nineteen days ago and failed on margin (4.06σ against a bar of 5)
rather than on direction. This is not a new problem and it is not being hidden —
but after 55 audits it should be said in one sentence: **the harness is
excellent and the creature is barely built.** 32% of everything demonstrated is
the harness measuring itself.

**3. A fourth consecutive week of free GPU quota will expire on Sunday**
(~28.7 h of 30 unspent), because both GPU cost classes are pilot-blocked on
measured evidence and the repair is a world redesign, not a dispatch. This is
`D15`'s subject and already before you; I record the fourth week rather than
re-escalate it.

**4. Nothing requires your attention on integrity.** Sections 1, 2, 6 and 7 are
clean on mechanical check. No threshold loosened, no control weakened, no
deadline moved, no decision acted on without record. Today's three spec results
were all honest FAILs with live controls — which is the ladder working, not the
ladder failing.
