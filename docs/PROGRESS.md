# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the anatomy audit and the completeness audit are Sunday
> work and were deliberately skipped; the last FULL page is 2026-08-31).

**2026-09-03 06:3x–07:0x UTC — DAILY.** Window: the last 24 h
(2026-09-02 06:3x → 2026-09-03 06:3x).

*The one sentence: **the builder spent its day buying back certificates it
already owned, and the one genuinely new thing it learned — that the playground
cannot make the sounds GOAL.md names — is the sixth independent instrument to
say the world, not the brain, is what is stopping this project.***

---

## The numbers

| | now | 09-02 06:3x (DAILY) | Δ |
|---|---|---|---|
| demonstrated / registered | **95 / 225** | 94 / 217 | **+1 / +8** |
| pass rate | **42.2%** | 43.3% | −1.1 pts |
| FAIL / VOID (live rows) | **22 / 10** | 20 / 10 | +2 / — |
| unreachable specs | **89 / 225 (40%)** | 85 / 217 | +4 (registration), floor now follows |
| rework (ledger rows at attempt > 1) | **97 / 127 = 76.4%** | 82 / 124 = 66.1% | **+10.3** |
| commits, last 24 h | **55** | 55 | — |
| builder slots fired | **25 starts, zero `PACING:` skips** | 25 | — |
| ledger settlements in window | **~86** (of which **2** are new science) | 13 | — |
| ratchets | coverage **rc=2** (4 CLAIM-DEAD, routed, DUE 09-11) · review-queue rc=0 (27 OPEN of 31) · champions rc=0 · decisions rc=0 | — | — |

**Read the settlement column before the pass column.** Eighty-six ledger
settlements landed in the window and **two of them are new science**: `HR.7`
PASS (the A2 hearing stem keeps bearing — worst-seed probe 0.9453 vs the 0.90
gate, mono null 0.1615 vs the 0.30 cap, swap inversion 0.9783) and `HR.5` FAIL
(pre-stated, and a real measurement of the world). Two more real negatives
opened the window — `T3.09` and `ME.11`. Everything else, on the order of
eighty rows, is the bounded-gate sweep re-stamping certificates plus re-buys.

**Four of those re-buys were repairing our own damage.** At 19:08 on 09-02 a
cross-organ doc-write race dirty-stamped `PS.01`, `PS.02`, `PS.03` and `BA.01`
into VOID; the following ten hours bought all four back at a cost including
`PS.01`'s 870.8 s and `BA.01`'s 227.7 s. That is what the rework rate moved
+10.3 points on — **the number is now measuring damage repair, not iteration
quality**, and it should be read that way until the 09-06 row lands.

**Pass rate fell while the count rose, and this time it is information, not
Goodhart.** The registry grew 217 → 225 because `HR.1`–`HR.8` were registered
from `HEARING_BAKEOFF.md` under the 5-step protocol, five of them deliberately
blocked behind same-family parents with the justification named in the ratchet's
growth log. A registration that refills an empty cost class (`cpu<10min` went
EMPTY-no-path-in → fillable) is the inventory repair the W34 post-mortem asked
for. The dilution is real and it was bought on purpose.

---

## The frontier, recomputed

`T2.01` — *"Locomotion beats a random policy"* — **blocks 38 specs**, four times
the next largest (`T4.04`, `T3.02`, `LT.01` at 9). It has been settled FAIL
since 2026-08-12 at 2.67σ against an unmoved 5σ bar. It is **not** a re-run and
the builder is right not to touch it.

Its repair path runs through `D1.0`, the control-path bakeoff that fired as D1's
armed default on 09-01, **burned 16.17 GPU-hours — 54% of a weekly quota — and
returned VOID.** The VOID is honest and the arithmetic says it is not
foreclosed: `c_e2e` returned 404.3 against random's 108.7, a 3.7× gain, and was
recorded as not having learned only because it was scored against its own wider
spread (2.56σ) while the other three arms were scored against random's. That is
a gate-scoring artifact, not an envelope wall.

**Three rows own this and their clocks are correct**:
`d10-learning-gate-uses-two-different-denominators` and
`d10-learning-gate-sits-at-the-untrained-twin-level` (both DUE **2026-09-06**,
mine), then `d10-successor-rerun-under-adopted-gate` (DUE 09-08). **W36 opens
2026-09-06 00:00 UTC with 30 free GPU-hours against attempt 1's measured
16.17 h** — after three consecutive weeks of expired quota, this is the first
week the largest unblock in the project has a named buyer.

**And the instrument cannot see any of it.** Nothing in the registry declares
`depends_on: D1.0`, so `run blocked` scores `D1.0`'s mass at **zero**. The
ranker shows the builder a 38-mass blocker that is settled FAIL and then the
trail to its actual repair goes cold *inside the tool*. The 60th audit had to
route that row by hand. See FOR THE OWNER.

---

## The honest paragraph

We did not get closer to a creature yesterday; we got closer to knowing why we
have not been. The day's real work was janitorial — a sweep re-stamping things
already proven, and four certificates bought twice because two of our own organs
wrote to the same documents at the same moment. But underneath the busyness one
thing genuinely happened: a brand-new sense family met the world for the first
time, and the world could not hold up its end. He cannot hear water because
water is a force field the apple never enters; he cannot hear a creak because a
creak is a placement onset; there is no label saying what a sound *is* and none
saying whether he made it. That failure was pre-stated, it was honest, and it
is the sixth instrument in a row — after the survival screen, the lookahead
test, the shelter oracle, the balance probes and the smell pilot — to return the
same verdict from a different direction. Smell, balance, shelter and warmth are
all now formally claim-dead, every claim spec parked or foreclosed, and not one
of them died because Jack failed to learn. They died because there was nothing
there to learn. The most important step toward Jack this week was building him
an ear that provably reports direction; the most concerning drift is that we
keep building him senses to point at a world with almost nothing in it, and our
instruments are getting very good at documenting that in ever finer detail. The
week's single decision that matters is not which gate to score `D1.0` with. It
is whether Sunday is spent on the world.

---

## REWRITTEN / STRENGTHENED

- **`experiments/coverage.py` — `UNREACHABLE_BASELINE` 90 → 89 (floor, not
  reading).** `HR.7`'s PASS reopened its downstream and the live count fell to
  89. Commit `b8f69f4`'s message says *"record unreachable 90 → 89"* and it wrote
  the new number into `ratchet_readings.json` — but the constant that the ratchet
  actually compares against was left at 90, so for the last half hour the shrink-
  only ratchet was carrying a floor one above the truth and would have accepted a
  silent regression back to 90 as clean. **Stronger because the floor now binds
  where it reads:** the ratchet fires on any regression, and the growth log now
  records shrinks as well as raises, so a floor that fails to follow the number
  down is visible in the file rather than only in a transient amber line.
  Verified: `run coverage` reads *"89 of 225, baseline 89"*, `stale_baseline`
  clear, no new red (the standing rc=2 is the 4 CLAIM-DEAD commitments, routed
  and DUE 09-11).

- **`scripts/ladder_prompt.md` — priority items 1 and 2 struck as DONE.** Both
  were finished work still being presented to the builder as the top of its list:
  `W0.DIAG` registered/run/**PASS** on 08-31 with the binding known-answer control
  cleared, and `T0.01`/`T0.10` both PASS with rows committed. Two further stale
  sentences corrected in place — `SM.03`'s orphaned file is tracked, and `SH.02`
  *"has no implementation at all"* is wrong (it has one; both are PILOT-BLOCKED
  with redesigns routed here). Replaced by live items **1'/2'/3'** naming
  `HR.6`'s CPU staging arms as the cheapest fresh unit, the standing **do not
  re-dispatch `D1.0`** order with its Sunday gate dependency, and the ranker
  blind spot above.

No spec files were touched and no threshold moved in either direction. Part 2 is
Sunday's.

---

## FOR THE BUILDER

1. **`HR.6`'s CPU staging arms (A0/A0b/A2/A5)** — minutes each, no download, and
   the registry's staging note means **if A2 cannot beat A0b on CPU the GPU arms
   are cancelled for free.** `experiments/hearing.py` is tied to `HR.7`'s
   certificate by IMPL_DEPS: editing it obliges a re-buy.
2. **Do not re-dispatch `D1.0`, and do not manufacture a substitute GPU job to
   spend W36.** Attempt 2 is gated on a gate design owed here on Sunday; an
   unchanged re-dispatch is a seed-lottery redraw. The quota has a buyer and the
   buyer has a date.
3. **`HR.1`–`HR.4` stay blocked-on-disk.** D19's default is NO-FETCH until
   09-14. Do not fetch a corpus to unblock a family.
4. Nothing else on this page needs you. The five rows due Sunday are mine, not
   yours; do not pre-empt them with a repair.

---

## FOR THE OWNER

1. **Sunday 2026-09-06 is oversubscribed, and I am telling you the order I will
   take it in rather than discovering it at turn 100.** Six OPEN queue rows come
   due that day — `w0-too-shallow` (the oldest live row, 10 days),
   `cross-organ-doc-race-voids-certificates`, `lt01-c2-body-cannot-rise`,
   `lc07-checkpoint-branch`, and the two `d10-*` gate rows — on the same run that
   owes Part 2, the anatomy audit and the completeness audit, in 40 minutes and
   120 turns. **My order: the two `d10-*` gate rows first** (they are a scoring
   choice, they are cheap, and they release a 16-hour dispatch into a quota that
   expires 09-13), **then `w0-too-shallow`**, then the rest, and Part 2 sampled at
   its minimum of 8 rather than dropping a dated row. If you want a different
   order, this is the page to say so on.

2. **The world is now the measured bottleneck on six independent instruments,
   and four constitutional commitments are formally claim-dead behind it** —
   smell, balance, shelter/building and thermal, every claim spec parked or
   foreclosed, none of them because Jack failed to learn. `HR.5` made it six this
   morning in a family one day old. **My recommendation: W1 stops being a queue
   row and becomes the project's stated stage.** We are at step 2 of GOAL.md's
   path building senses for a step-6 world, and the ladder is measuring the gap
   very precisely. This is the strategic fork; the `D1.0` gate is a detail beside
   it.

3. **`run blocked` cannot see the project's largest unblock.** `T2.01` blocks 38
   specs; its repair runs through `D1.0`; no spec declares `depends_on: D1.0`, so
   the ranker scores that edge at zero and the 60th audit had to route the work by
   hand. **My recommendation: do NOT add the edge to the registry** — it would
   make `T2.01` unreachable until `D1.0` passes and would drift its certificate.
   Instead the ranker should read a declared `repaired_by` field that carries mass
   without carrying blocking semantics. That is a real design change to
   `run blocked`, so it is yours to authorise, not mine to make.

4. **Organ liveness, all green.** builder 06:07 (hourly, 25/25 slots, no
   `PACING:` streak), overseer 06:37 (6 h), field watch 08-31 05:53 (Mondays —
   next fire 09-07, inside cadence, wk5 consumed 08-31), review 06:37 (this run).
   No organ is silent.
