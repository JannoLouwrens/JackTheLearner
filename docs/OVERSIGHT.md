# OVERSIGHT — 56th audit, 2026-09-01 00:40 UTC (HEAD `8ef92a4`, tree clean)

## VERDICT: DRIFTING — the ladder is honest and the eleven firings were lawful, but the Review's liveness watch went blind on the exact night the Review first succeeded, and it committed the falsehood to the repo

Sections 1, 2, 6 and 7 have **no findings**, checked mechanically. Saying so
plainly is the most valuable result in this report, because it is what makes
finding 1 worth taking seriously rather than reading as noise.

---

## FINDING 1 (top rank) — `review_liveness` cannot see a `**FULL**` row. The alarm is stuck ON, and it stamped a false banner onto the freshest document in the repo.

**The Review completed its first-ever FULL run on 2026-08-31** — commit
`9d0908e`, 684 lines rewritten in `docs/PROGRESS.md`, twelve generality
barriers, and it **did** write its history row (`docs/PROGRESS_LOG.md:19`).

It wrote the mode in bold, because it was the first one ever:

    | 2026-08-31 | **FULL** | 94/211 | 44.5% | +1 (93→94) | 66.1% | **THE FIRST FULL RUN…

`scripts/lib_liveness.sh:77` matches on exact string equality:

    history_newest_mode_date() { _md_table_rows "$1" | awk -v m="$2" '$2 == m {print $1}' | ... }

Field 2 is `**FULL**`. It is not `FULL`. Verified live in the repo's own shell:

    FULL  -> []              # empty — "that mode has never completed"
    DAILY -> [2026-08-31]
    grep -c "| FULL |"  -> 0
    grep -c "| DAILY |" -> 12

**The concrete damage, in three parts:**

1. **A false statement is now committed to the repo.** `8ef92a4` (today,
   00:37) stamped `docs/PROGRESS.md` with:
   *"STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING… no FULL
   row has EVER been written… that mode has never completed."*
   That banner sits **three lines above** the page's own first sentence:
   *"2026-08-31 — FULL, and it is the FIRST FULL RUN THIS PROJECT HAS EVER
   COMPLETED."* The document refutes its own banner on sight.
2. **It tells the builder to distrust the FULL analysis.** `PROGRESS.md` is
   where FOR THE BUILDER redesigns live and the builder reads it every
   iteration. The banner instructs it to read all 684 lines as "a RECORD, not
   current state" — including the twelve-barrier finding and the ordered work
   list the builder is currently executing.
3. **The alarm is now permanently on, so it carries zero information.** This
   watch exists (52nd audit) precisely because the Review's failures were
   invisible. It fired correctly for three dead Sundays. On the first Sunday it
   worked, the instrument could not see the success — and it will now fire
   identically forever, whether the Review runs or not. A constant is not a
   signal. **The next genuinely missed Sunday will look exactly like tonight.**

This is the sharper half: the failure is in the *false-alarm* direction, which
this repo's design philosophy ("fail toward more oversight") reads as safe. It
is not safe here. It destroyed the alarm's discriminating power and produced a
wrong write.

**Not a papered-over caveat elsewhere — a real bug.** `table_liveness` reports
`newest` row age correctly (the date field parses fine); only the *mode* field
is compared strictly.

---

## FINDING 2 — the eleven armed defaults fired lawfully. I checked the clause SYSTEM.md says nobody checks.

`SYSTEM.md:145-154` states plainly that two of the three safety clauses on a
default — *no `GOAL.md` edit, no weakened threshold* — "are properties of the
**commit that fires**… They remain on the author's word." Eleven fired in one
18-minute iteration (`8b17d41`…`f65f1d4`), each hand-asserting its own
compliance in prose. That is the exact author-self-certification law 1 exists to
distrust, so I audited the firing diffs directly rather than the arming text.

**Result: clean.** Mechanically, across all eleven commits:

- **Zero** touched `GOAL.md`. **Zero** touched `SYSTEM.md`. **Zero** touched
  `experiments/ledger.json`.
- Five touched registry files. Every numeric change is an **addition** or a
  **tightening**: D1.0's new 3σ learning gate and 1.5σ margin, LC.07's
  `>=3 sigma` twin gate and `n_lives >= 12`, D12's transcribed convergence
  guards (`slope <= 0`, crossover beyond 3×). No constant moved in the
  loosening direction; no `_check` gained an `or`; no seed count fell.
- The one deletion that looks alarming in a diff —
  `- depends_on=["BA.01"]` in D8's `75fd1e0` — is a reformat, not a removal.
  Live value is `['BA.01', 'LT.08']`: BA.01 retained, LT.08 **added**. A
  strictly-narrowing park, exactly as the commit message claims.
- D13's condition (3), the one that could silently disable the new overseer
  no-op, is **sound**. Its em-dash grep matches `experiments/decisions.py:497`
  byte-for-byte, and the live `due` dates parse (`2026-09-05` ×2,
  `2026-09-07`). I verified this rather than accepting the commit's claim,
  because a broken condition (3) would let the audit skip a slot on the day a
  default is due.
- The no-op is currently **inert and correctly so**: `overseer_noop.state` does
  not exist, so `noop_eligible` returns 1 and this audit ran in full. It fails
  toward more oversight as designed.

**One honest correction already on the record:** `75fd1e0`'s message asserts
`coverage rc=0`. It was rc=2. The builder caught this itself and said so in
`f34251a` ("a pipeline-tail misread"). I hit the identical trap twice in this
audit — `cmd | tail; echo $?` reports the *pager's* status. It is a recurring
hazard in this repo and it is now a lesson.

---

## FINDING 3 — coverage is rc=2, it is genuinely inherited, and the firing iteration made it one class worse

The builder's claim ("inherited, not mine") is **true, and I verified it** by
checking out the pre-firing commit `b4805ac` in a worktree: rc=2 there too.

But the framing is incomplete. The red condition is `new_empty_class`, and the
set grew across the firing iteration:

    pre-firing  (b4805ac): 3 classes — cpu<48h, gpu<20min, gpu<2h
    now         (8ef92a4): 4 classes — cpu<2h, cpu<48h, gpu<20min, gpu<2h

`cpu<2h` held exactly one row — `BA.02` — and **D8's firing parked it**. The
class is now EMPTY. This is not a violation (parking is permitted, nothing was
weakened, and BA.02 was VOID rather than a fresh dispatch), but "inherited, not
mine" understates it: the iteration added a class to the red set it was
reporting as pre-existing.

The constitutional check is **green**: `0 commitments with NO declared spec, 0
CLAIM-DEAD`. Nothing in GOAL.md is unfalsifiable. `cpu<2h` is fillable today
(LG.02, T3.09, UB.14).

---

## FINDING 4 — compute: ~79 free GPU-hours expired unused in three weeks, and W36 is at zero

    2026-W33: used  7.89 of 30 h  ->  22.11 h EXPIRED
    2026-W34: used  1.62 of 30 h  ->  28.38 h EXPIRED
    2026-W35: used  1.28 of 30 h  ->  28.72 h EXPIRED
    2026-W36: 0 jobs so far (week began Mon 2026-08-31)

Waste from *failed* jobs is negligible — 8 not-ok jobs totalling **1.43 h**
lifetime. The loss is not spent-badly, it is **never-spent**: both GPU cost
classes read NOT FILLABLE for most of that period, so the quota was
structurally unspendable however awake the loop was.

**That changed last night.** D10 and D1 registered `LC.07` and `D1.0`, and
`gpu<8h` has dispatchable inventory for the first time since T2.15 was consumed
on 08-25. Neither has a test file yet, so the refill is real but not yet
spendable — and W36's 30 hours are running.

---

## Sections with no findings (checked, not assumed)

**1. Ledger integrity — clean.** 93 PASS entries. **0** name a commit git has
lost (`git cat-file -e` on every distinct commit). **93 of 93** whose spec
declares a control carry non-empty `control_metrics`; **0** declare no control.
**0** PASS specs lack an implementation file (128 files in `experiments/tests/`).

**2. Thresholds over time — no silent loosening.** Full `git log -p` over 7 days
on `registry.py`, `registry_expansion.py`, `tests/`. Every numeric movement is a
new gate or a strengthening. The notable one is **T2.10**, which moved the
*other* way on purpose: `468772e` added `MIN_PARA_MARGIN = 0.10` as a
**conjunct** (`_check` gained two `and` clauses, not an `or`), and `b4805ac`
then re-bought the spec **honestly RED** — the PASS→FAIL that took the count
94→93. A spec voluntarily made harder and failed on its own new clause is the
system working exactly as designed.

**6. Stuck decisions — clean.** `decisions --check` ratchet ok, 0/10 undeclared.
No `MEANS-ESCALATED`, no `OVERDUE`. Three armed and future-dated (D15/D16 due
09-05, D17 due 09-07). All eleven fired entries are recorded in
`DECISIONS_RESOLVED.md` with losers named and reversal paths stated. Nothing was
acted on without being recorded.

**7. Bakeoff hygiene — one item worth naming, and it is disclosed.** D10 seats
`wm-latent` **BY VERDICT** off a screen that returned exactly one learner. A
single-arm race is not an arbitration in the usual sense, and "two non-learners
cannot arbitrate" is this project's own rule. It is **not** a VOID-treated-as-
verdict: LC.03 stays CONCLUDED-VOID in the ledger, the caveat is written on the
seat's face, adoption remains gated behind unison, and `LC.07` was registered in
the same commit as the challenger. Disclosed, contestable, and reversible — the
correct handling of a weak verdict, not a hidden one.

**3/4. Builder — alive, honest, and mostly not building Jack.** 24 iterations in
24 h, **24 rc=0**, no credit exhaustion, no repeated identical failures, no
paused loop. One leftover process on 08-31 19:12 (a 1-second
`experiments.decisions --check`), correctly named by `lib_procwatch` and since
gone. PASS delta over 24 h: **93 → 93** (94 at peak, one honestly removed by
T2.10). Of the 24 iterations, roughly **12** went to governance — the eleven
defaults, the journal, the queue and champions bookkeeping. Three produced
registered-spec numbers about Jack (LT.01, T2.10, ME.11 E/F).

---

## FOR THE BUILDER

**B1 (do this first — it is a two-character fix with a large blast radius).**
`scripts/lib_liveness.sh:77`, `history_newest_mode_date`: the mode comparison
`$2 == m` cannot see `**FULL**`. Strip markdown emphasis before comparing —
normalise field 2 by removing `*` and surrounding whitespace, then compare.
Do **not** repair this by editing the `**FULL**` row in `PROGRESS_LOG.md`: the
row is a truthful historical record and the instrument is what is wrong. After
the fix, `review_liveness` must print `OK — 2026-08-31 daily, 2026-08-31 FULL`.
Then remove the false STALE banner from `docs/PROGRESS.md` (added by `8ef92a4`)
and say in the commit message that the run it accused of producing nothing
produced 684 lines.

**B2 (the guard that makes B1's class of bug unrepeatable).** The instrument
failed *silently and in the direction that looks safe*. Add a known-answer
fixture to `lib_liveness.sh` asserting `history_newest_mode_date` returns the
date for all of `FULL`, `**FULL**`, ` FULL `, and `_FULL_`. A liveness watch
with no test for its own matcher is the T0.31 disease one organ over: a ratchet
whose wiring is unasserted can be disconnected by a formatting choice.

**B3.** `gpu<8h` has inventory for the first time since 08-25 — `LC.07` and
`D1.0`, both registered, both needing test files, W36's 30 free hours running
and 0 spent. This is the single highest-value use of a GPU-capable iteration.
`D1.0` is the one that unblocks 38 specs.

**B4.** When reporting a ratchet's exit code, never read it through a pipe.
`cmd | tail; echo $?` reports the pager. Use `cmd > /tmp/out; rc=$?`. This has
now cost the builder one false `rc=0` in a commit message (`75fd1e0`) and cost
me two misreads in this audit before I caught it.

---

## FOR THE OWNER

**Nothing here needs a decision from you.** Three armed defaults remain with
future dates (D15/D16 on 09-05, D17 on 09-07); the eleven that fired overnight
did so lawfully — I checked the firing diffs against the two safety clauses
`SYSTEM.md` admits nobody enforces, and they hold. No threshold was weakened, no
constitutional file touched.

**The one sentence that matters, and it has not changed in 56 audits:** the
harness is excellent and the creature is barely built. **Tier 5 — the thesis —
is 2 passing specs of 45.** The ladder-and-apple standard, the thing GOAL.md
opens with, is the `LT` family: `LT.01` FAIL, and `LT.02`–`LT.09` all NOT_RUN.

Last night was a good night for the *machine*: twenty-two days of D1 deadlock
broken, four architectural seats made contestable, the GPU queue refilled. It
produced **zero** new demonstrated capability, and net **−1** — because a spec
was made harder and honestly failed. I want to be clear that I count that
correctly: T2.10 going red is the system working. But twelve of twenty-four
iterations went to governance, and the answer to "closer to a curious humanoid,
or just a longer list of green ticks?" is: **neither, today — closer to a
machine that can finally start.** The unblocking was real and it was overdue.
The test of whether it was worth it is whether `D1.0` and `LC.07` get
implemented and dispatched this week, into a 30-hour quota that has expired
nearly untouched three weeks running.

And one uncomfortable note about my own organ: the instrument that watches the
Review is the *fourth* oversight instrument in eight days found to be reporting
a number it could not actually support (after `coverage.py`'s exit wiring,
`decisions.py`'s unenforced clauses, and `run blocked`'s foreclosure blindness).
Every one was caught by reading the instrument rather than its output. That is
the pattern worth funding.
