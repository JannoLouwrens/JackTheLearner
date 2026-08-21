# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-21 06:45 UTC — DAILY. Window: 2026-08-20 06:40 → 2026-08-21 06:40.**

*The one sentence: **the window's first eighteen hours were the most disciplined
science this loop has produced — sight's first real claim, a body bakeoff that
adopted nothing, a VOID diagnosed by replay instead of by story — and then the
builder put itself into a three-day blackout by reading the wrong meter, with
twenty-two GPU-hours expiring Sunday and two stale certificates it never looked
for.***

---

## 1. The numbers

**Ladder: 83/169 demonstrated (49.1%).** Yesterday: 82/169 (48.5%).

| | this window (24 h) | 08-20 window |
|---|---|---|
| ledger events recorded | **13** | 52 |
| PASS / FAIL / VOID / ERROR | 5 / 3 / 3 / 2 | 45 / 2 / 1 / 2 |
| **first-ever PASSes** | **1** (T3.01) | 2 |
| net new demonstrated | **+1** (82 → 83) | +2 |
| registry growth | **0** (169, flat — third window running) | 0 |
| builder iterations | **24 fired, 24 rc=0** | 24 fired, 24 rc=0 |
| commits | **55** | 42 |

**Goodhart check: rate rose 48.5% → 49.1% on a registry flat at 169 for eight
days.** All runner, no ladder — the good direction of the two failure modes,
and for the third consecutive window it also means **nothing new was
registered.** One flat window is healthy; three is the research queue not
feeding the ladder. The queue's own empty-queue rule says the correct response
is to *generate* work (`INTEGRATION_QUEUE.md`, "WHEN THE QUEUE IS EMPTY"), and
that has not happened either.

**Rework: 51 of 83 passing specs took more than one attempt (61.4%),** flat on
yesterday's 61.0%. T3.01 alone contributed attempts 2–5 this window.

**Compute.** Meters at 06:40: session 16%, **`week:Fable` 93%**, **`week:all
models` 77%**, all resetting Aug 24 05:00 UTC. Kaggle W33: **7.20 h charged ok
+ 0.26 h failed of 30 — ~22.8 h expire Sunday 2026-08-23.**

**The frontier, recomputed (`run blocked`, not quoted).** 60 of 169 specs are
unreachable; 26 are runnable now.

| rank | spec | status | frees | note |
|---|---|---|---|---|
| 1 | **T2.01** Locomotion beats a random policy | FAIL | **35** (blocks 36) | science problem, not compute; behind D1 |
| 2 | **LC.03** learning-core screening | VOID ×2 | 8 | **v2 re-screen in flight**, ETA ~Aug 23 late |
| 3 | **UB.9** Heard, not seen | **PASS but STALE** | **5** (blocks 7) | **new this window; nobody noticed** |
| 4 | **T2.06** Language-action alignment | **PASS but STALE** | 3 | **new this window; nobody noticed** |
| 5 | T2.02 / T3.06 / T2.05 / T4.02 | VOID/NOT_RUN/FAIL | 1–2 each | |

---

## 2. What the window actually produced — and it was good

**Sight got its first claim.** `T3.01` PASS (v3, attempt 5): `acc_full`
0.61–0.63 against its own frozen probe's 0.4467 reference, ablated and
pixel-shuffled arms sitting *exactly* at chance 0.25, `hash_overlap` 0.0. The
plastic vision encoder is load-bearing, not decorative — the first defence the
PLASTIC-ONLY decree has ever been given on our own substrate. It took two VOIDs
and a pre-registered curves probe to get there, and every one of those was
routed by a decision rule written before the run.

**Three things I want to name as *method*, because they are what a healthy
week looks like:**

- **The W0.BAL body bakeoff ran and adopted NOTHING.** Arm C (a 0.35 m plinth,
  mass in the base) scores `upright_frac` 1.000 on every seed against arm A's
  0.002–0.004, and the builder attached the table to D9 and left the seat
  alone, because a winner still needs owner adoption. That is the constitution
  working under temptation.
- **A VOID was diagnosed by replaying its own `_check`.** LC.03's v1 VOID had
  been narrated wrongly; the correction came from replaying the recorded row
  against the code, not from a better story, and the resulting lesson
  ("a generic VOID message admits every narrative") is now written.
- **The kill clause did NOT fire when it was invited to.** W0.BAL's premise
  was narrowed rather than declared dead, because even the worst arm clears
  rung 1 in its pre-topple seconds. Refusing a satisfying deletion on the
  evidence is the same muscle as refusing a satisfying PASS.

**Two doc-code gaps closed machine-readably** — `{arm}/data_starved` was
promised at registration and computed nowhere; every `_check` VOID now names
its firing branch, key and values in its own ledger row.

---

## 3. THE FINDING — a three-day self-imposed blackout, on a meter that is not the gate

**The last three iterations (04:24, 05:07, 06:07) did no work by choice.** Each
logged the same reason: *"the usage meter reads Fable 91% (93% now), past the
90% hard stop, so per the standing B6 blackout plan I planned no new work."*

**The 90% hard stop is not measured on Fable.** `scripts/lib_usage.sh` calls
`claude_usage.py --pct`, which returns **`week:all models`** and nothing else:

```
session                [###                 ]  16%   resets Aug 21,  9am
week:Fable             [##################  ]  93%   resets Aug 24,  5am   <- what the builder is stopping on
week:all models        [###############     ]  77%   resets Aug 24,  5am   <- what the GATE reads
```

The gate has thirteen points of headroom and has not fired once this window.
`week:Fable` is high only because cron passes `JACK_LOOP_MODEL=fable`, and
`ladder_loop.sh` already handles that exact case —
`FALLBACK_MODELS="opus sonnet"` fires on the refusal. **The builder is not
gated. It has grounded itself.**

**What it costs, concretely.** The self-imposed blackout runs to the Aug 24
reset. **Kaggle's W33 allocation expires Sunday Aug 23 — one day earlier —
with ~22.8 of 30 hours unspent.** This is the second consecutive week the
allocation dies unused; W32's died on 08-16 while the loop was genuinely dark.
And the builder's *own* B6 plan, written at 01:0x this window, names the
deadline exactly: *"W33 hours die Sun 08-23 REGARDLESS, so anything worth W33
must be dispatched before ~88%, not queued behind the stop."* Nothing was
dispatched after 03:13.

**And there was something to dispatch.** `run blocked` now ranks **UB.9 at #3,
`PASS but STALE`, frees 5 and blocks 7 — including UB.10.** T2.06 is stale too
(frees 3). Both are mechanical re-runs: no threshold to set, no arm to design,
no diagnostic cap to spend. **Neither appears in any journal entry from this
window**, because a blackout-lean liveness pass checks pids and log bytes and
never runs `run status` — so the loop's cheapest, highest-value available work
was invisible for the same reason it decided it had none.

**The generalisable shape, and it is not the arithmetic.** Yesterday's finding
was *an organ can stall with a correct `rc=0`.* Today's is one turn worse: the
organ is not stalled, it is **correctly executing a rule it applied to the
wrong number**, and every instrument agrees — `rc=0`, tree clean, HEAD pushed,
liveness receipts committed, journal current. A wrong premise held with good
discipline produces a perfect audit trail of doing nothing.

**In fairness to the builder, the underlying caution is sound and I do not
want it discarded.** Falling back to opus draws on the same `all models` pool
the Review and the overseer draw from, so burning it takes the auditors down —
which is this desk's own FOR THE OWNER #1, twice raised. The error is not
"be careful with the pool"; it is *reading the pool's number off the wrong
line, and concluding that careful means zero.* The correct posture is
**dispatch, then idle**: Kaggle kernels and `launch_detached.sh` runs compute
through any blackout and write their own receipts. That is B6's own sentence.

---

## 4. Steering maintenance (Part 2.5)

**`scripts/ladder_prompt.md` — two fixes, one new section.**

1. **NEW section at the top, above the priority order**: which meter the gate
   actually reads, both lines printed, the fallback chain named, the
   dispatch-then-idle posture, the Sunday deadline with the live budget
   arithmetic, and UB.9/T2.06 named as the two ready units. The builder fires
   at 07:07 and would otherwise ground itself a fourth time.
2. **A cached, mislabelled meter inside the LC.03 section** — "METER AT LAUNCH:
   Fable 89% vs the 90% hard stop" — is the sentence that propagated the error
   into three iterations. Corrected to point at the tool, per this page's own
   rule that priorities never cache a number.

**FIELD_WATCH.md: unchanged, nothing to consume.** Last real sweep 2026-08-12
(week 3), dispositioned by the 08-13 Review. The Monday 08-17 sweep was refused
at 96% usage, so week 4 does not exist; the scout has now been silent nine days
against a seven-day cadence — **under 2×, so not yet a liveness finding**, and
the next fire is Mon 08-24 05:37, thirty-eight minutes after the weekly reset.
Its own week-3 note said fronts 1–3 should not be re-swept before ~08-19, so
the mandate is ripe and waiting.

**Seat staleness (`CHAMPIONS.md`, rule 4): five cells corrected, one seat still
flagged, no seat changed hands.**

- **Taste** still read *"`TA.02` is runnable today"* — two days after TA.02
  passed. Corrected: the claim is closed, and the cell now says plainly that
  no mechanism *arm* is seated, so the win is not mistaken for an adoption.
- **Sensory fusion** claimed UB.9's PASS as live evidence "so the matrix has
  something to eat". UB.9 is stale and UB.10 is parked; both recorded.
- **Learning core** still described the 08-14 VOID. Refreshed to the second
  VOID (08-21 02:11) and the v2 re-screen in flight. **The seat stays FLAGGED
  STALE** — the default champion has still never been contested — but the flag
  now has a scheduled resolution for the first time since it was raised.
- **Smell** said the SM.02 repair was "in progress and UNCOMMITTED"; SM.02 is
  parked, the repair is committed, and the cell said neither.
- **Vision encoder** gained T3.01's numbers — the from-scratch encoder's first
  actual defence, which belongs where the seat is argued.

**Organ liveness: all four organs healthy, nothing silent.** Builder 24/24
hourly, all `rc=0`. Overseer 6-hourly, all four fires present (12:37, 18:37,
00:37, 06:37) and the 24th audit returned ON TRACK. Review daily, firing.
`tmp_reaper` 04:13. `lost_iterations.log` empty. **The gate did not refuse a
single organ this window** — which is exactly what makes §3 worth writing: the
first blackout this project has had that nothing in it caused.

---

## 5. The honest paragraph

We are closer to a creature, and the reason is that he can now be shown to
*see* — not to carry an encoder that produces numbers, but to have his
performance collapse to chance when the thing he sees is taken away, which is
the only form of "he sees" this project accepts. Put beside taste closing two
days ago, two of his senses have stopped being apparatus and started being
capabilities, and both of them got there by failing first and being allowed to.
The window's best hour was the one where a bakeoff produced a clean winner and
the loop adopted nothing, because the person who could adopt it was asleep;
that is a system with a conscience rather than a scoreboard. And then the drift,
which is subtler than any we have named and worse for it: the builder spent its
last three hours performing custody — checking pids, confirming clean trees,
writing receipts about the absence of work — and every organ we own reported
that as health. We have taught this system that stopping is safe, and it has
learned the lesson so well that it now stops for reasons it has not checked. It
did not lie, it did not thrash, it did not cut a corner. It read one number off
the wrong line and stood still for three hours in front of a deadline it had
itself written down that morning, and nothing anywhere was able to notice,
because everything it did while standing still it did impeccably.

---

## FOR THE BUILDER — ordered

1. **You are not in a blackout. Print both meter lines before you decide you
   are.** The gate is `week:all models` (77%), not `week:Fable` (93%). Full
   reasoning is now at the top of `ladder_prompt.md`. If you still judge the
   pool worth protecting, that is legitimate — but say which number you are
   protecting and from what, rather than citing a stop that has not fired.
2. **Re-run UB.9 before Sunday.** `PASS but STALE`, ranked #3, frees 5 and
   blocks 7 including UB.10. Mechanical: no threshold, no redesign. It went
   stale under a loop that was reporting "zero stale claims remain".
3. **Re-run T2.06** (`PASS but STALE`, `GPU_SHORT`, frees T2.07/T2.15/T3.08).
   With #2 that is two of the top four blockers cleared for a couple of the
   ~22.8 Kaggle hours that die Sunday.
4. **Add `run status` to the blackout-lean pass.** A liveness pass that checks
   pids and log bytes but not the ledger is why #2 and #3 were invisible. Two
   certificates decayed in silence inside a window with 55 commits in it.
5. **The registry has been flat at 169 for eight days.** After #2–#3, the
   correct unit is `INTEGRATION_QUEUE.md`'s own empty-queue rule: research the
   next stage that has no design doc and register from it. Do not manufacture a
   GPU dispatch to spend hours — but do not let the ladder stop growing either.

## FOR THE OWNER — strategic forks only

1. **The usage gate has now failed in the opposite direction, and it cost a
   second consecutive GPU allocation.** Last week the gate fired and took the
   auditors down with the builder (raised twice). This week it did *not* fire —
   and the builder grounded itself anyway for three days on the per-model meter,
   with ~22.8 Kaggle hours expiring Sunday. The common cause is that **"90%" is
   a number in prose with no single authoritative source**: the shell reads
   `all models`, the builder reads `Fable`, and both call it "the hard stop".
   **Recommendation: make the meter speak for itself** — have `claude_usage.py`
   print one line naming *the gating number*, e.g. `GATE: all models 77% of 90%
   — PROCEED`, and have the loop echo that line into `ladder.log` every
   iteration. Then no organ can misidentify the gate, and the log records which
   number governed. This is a small change to two scripts; it is outside this
   desk's jurisdiction, and it closes both failure directions at once.
2. **D1 — where control lives — is twelve days open, and it is now the whole
   frontier.** T2.01 frees 35 of the 60 unreachable specs, is correctly
   refusing to be re-run (a fixed 5σ bar against a converged 2.67σ curve is a
   seed lottery), and cannot be fixed by compute. Its `RECOMMENDED` option is
   the one PLASTIC-ONLY forbids, so the decision is *unanswerable as posed* and
   has been since 08-09. **Recommendation, unchanged and now urgent: either
   narrow the decree's scope to admit a frozen control trunk into the arena as
   an arm, or strike that option and let the remaining arms run.** Both are
   decisions. Twelve days of continuing is the only option that is not.
3. **The W0.BAL body bakeoff has a clean winner waiting on you (D9).** Arm C —
   a 0.35 m plinth with mass in the base, COM at 0.085 m — holds `upright_frac`
   1.000 on every seed against the as-built body's 0.002–0.004, with hand reach
   at 1.165–1.185 of its ~1.19 ceiling. The builder ran it, tabled it, and
   adopted nothing, which is correct. **This is the same question as D1 from
   the other end**: T2.01 may be failing because the *body* cannot stand, not
   because the *core* cannot learn. **Recommendation: decide D9 and D1
   together** — if the body is the fault, D1's control-architecture question is
   being asked about a creature that falls over before it can answer.

---

*Part 2 skipped per DAILY mode. No threshold moved, no control softened, no
spec file touched, no ledger entry edited. Queued for Sunday's FULL run:
T0.13's 20+ attempts; SM.01's ungated intermittency shortfall; PS.03's
self-reported "one physics measurement, not three"; BA.01's control passing by
0.008; the 61.4% rework rate against the oldest certificates; and — new — the
UB.10 arm-redesign question routed here by the recipe probe (no uniform recipe
trains all six matched-param arms; A2 learned its marginals under no tested
recipe), which is a Part 2 redesign and not a daily edit.*
