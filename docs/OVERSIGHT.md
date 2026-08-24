# OVERSIGHT — 27th audit, 2026-08-24 18:45 UTC

## VERDICT: ON TRACK — the ledger is clean; the losses are compute and routing, not integrity

Nothing in the ledger moved in the wrong direction today. `run verify` re-judged
every auditable PASS from the record alone and returned **zero** failures on all
five probes. Not one threshold moved in the loosening direction in seven days —
the single control-direction change (T2.05) was loud, measured, pre-registered
against an expected FAIL, and paired with a strictly harder claim gate. All
three ratchets held: `coverage` exit 0, `decisions` 10/10 undeclared (unchanged),
`champions` 8/8 phantom arenas (unchanged).

Two things are wrong, and neither is the science.

**RANK 1 — the 26th audit's B6 was inverted, and W34's GPU quota is at zero on a
Monday evening.** B6 asked for *dispatch-then-idle*: when the Claude meter runs
hot, spend one lean iteration pushing detached Kaggle work that computes through
the blackout. What landed instead (`e03693d`) is a **pace gate that skips whole
iterations**. `gpu_budget.json` has **no `2026-W34` key at all**; the last GPU job
was `1787300777` at **2026-08-21T08:26 — 3 days 10 hours ago**; 30.0 free Kaggle
hours sit unspent and the loop is now dark by its own gate.

**RANK 2 — "routed to the weekly Review" is an unregistered backlog, and the
Review's last run died.** Three world-design forks — one of them a live physics
bug in W0 — are routed to an organ that has no queue file, and whose most recent
invocation exited `rc=1` on `API Error: 529 Overloaded` at 06:45 today with no
retry. A routed finding has no id, appears in no ranking, blocks nothing and
fails no gate. That is the same invisibility class `coverage.py` and
`champions.py` were built for, one level up.

---

## 0. Is the ladder the right ladder? — coverage clean, 14th straight audit

`experiments.coverage` exits 0. **0 of 23 constitutional commitments has no
declared spec.** The 2026-08-10 miss stays closed.

**14 commitments have specs but zero credited passes.** Only `claim`-kind
declarations are credited; `fixture`/`sensor`/`rule` are support. The 9
commitments that do have a credited claim-kind PASS each have **exactly one**:

| commitment | its only claim-kind PASS |
|---|---|
| sight | T3.01 |
| hearing | UB.9 |
| taste | TA.02 |
| damage/nociception | PS.03 |
| memory across lives | ME.10 |
| language (parent) | T2.06 |
| curiosity | T2.08 *(Tier 2, "curiosity drives coverage")* |
| generality | T1.02 |
| one brain / unison | UB.9 |

Two facts worth keeping in view. **UB.9 carries two commitments by itself** —
hearing and the 22-spec unison family — and its own registry notes now state
(correctly, 71c879f) that its at-chance unimodal readings rest on the
shared-trainer argument alone. **Curiosity's single pass is T2.08**, a Tier 2
component test; CU.1–CU.7, the family that would show the ladder-and-apple
sentence, are all unrun and behind D1.

---

## 1. Integrity of the ledger — clean, no findings

84 PASS / 8 FAIL / 3 VOID across 95 entries (+ DP.05's uncommitted FAIL, §5).

- **Implementations exist** for all 84 PASS.
- **Commits resolve**: every PASS `commit` field is a live object in git. `git
  cat-file -e` on all 84 → 0 failures.
- **Zero dirty stamps** among PASS entries.
- **Controls**: `run verify` reports `0` verdicts that no longer re-derive, `0`
  gates that ignore their control, `0` controls declared but never run, `0` gates
  that could not be replayed, `0` entries unauditable.
- **2 PASSes with no control at all** — `T0.01`, `T0.10`. Both Tier 0 harness
  existence claims; declared, unchanged, already carried in §1.2 of prior
  reports. Not a new finding.
- **28 entries predate `impl_sha`**: 27 verified byte-identical by git, 1 stale
  by content — `T2.02`, which is **VOID**, not a claim. Low priority.

Nothing here damages the trustworthiness of the ledger.

## 2. Thresholds and controls over seven days — no silent loosening

21 commits touched `registry.py` / `registry_expansion.py` / `experiments/tests/`
(+8,529 / −111 lines). Every deletion was inspected. Findings:

**Examined and cleared — T2.05's control direction changed, loudly and with
numbers.** The control gate moved from `mse_shuffled >= mse_persist` to
`mse_shuffled >= CTRL_TOL * mse_null` with `CTRL_TOL = 0.98` and
`mse_null = min(persist, mean)`. On seed 0 that lowers the VOID trigger from
1.092 to 0.808, and the 08-14 shuffled reading (0.824) that VOIDed under v1 would
now clear it. **This is defensible and was not hidden**: the 08-14 run *measured*
persistence uninformative at K=5 (persist 1.092/1.128/1.187 vs mean
0.824/0.860/0.914), so the shuffled arm was learning marginal statistics, not
leaking — and the redesigned null now owns those statistics. The 2% tolerance is
the "bar finer than one quantum" lesson applied to a genuine tie (0.824 vs 0.824).
Meanwhile the **claim** side was strengthened twice: the 20% margin now sits on
`min()` rather than the weaker `persist`, and a new absolute gate was added
(`mse_wm <= mse_ridge` every seed). The commit **pre-declares the expected FAIL**
from the 08-14 numbers (wm 0.178–0.231 vs ridge 0.114–0.131). Registry
`null_baseline` and `falsified_by` were updated in the same commit. That is the
correct way to move a control.

**Strengthenings, for the record:** LC.03 v2 quadrupled the envelope
(`N_STEPS` 100k→400k, `W_CLOCK` 4,320→17,280, `HALF_STEPS` 50k→200k) with
`SIGMA_GATE 3.0` explicitly **unmoved**; T2.03's `COVERS:` demoted `sight
(claim)` → `(fixture)`; T2.04 dropped a `COVERS:` marker that was never a
commitment name. All three *remove* credit rather than adding it.

**No seed count reduced. No assertion removed. No `_check` gained an `or`.**

## 3. Drift from the goal — none; every unit traces to a GOAL sentence

| unit (last 24 h) | GOAL.md sentence served |
|---|---|
| NE.00 PASS — homeostatic reward algebra | "hunger, thirst, temperature, pain… he must *live* in it" |
| NE.01 attempt 3 — FAIL, clean stamp, routed | needs as a real control problem; learning-by-living |
| BA.02 attempt 3 — VOID, re-certified on drifted `w0.py` | balance / "he feels himself falling" |
| DP.05 implemented + run — FAIL | fast/slow; "he must try, fall, and learn from falling" |
| `champions.py` arena-existence check | ARCHITECTURE always contested (owner ruling, 2026-08-24) |
| pacing gate (`e03693d`) | none directly — infrastructure; see §5 |

No drift. Two items deserve explicit credit because they are the *opposite* of
optimism, and this section usually only catches the reverse:

- **The DP.00 correction** (`4f5e458`): DP.00 — *"this world rewards looking
  ahead"* — PASSED in a 12×12 tabular gridworld, not W0, and the reading had
  drifted to "Jack's world rewards deliberation" and been repeated to the owner
  in that form. The builder caught its own certificate carrying a world it did
  not earn. **A PASS carries the world it was earned in, and a spec id does not.**
- **DP.05's fidelity pilot** found two real W0 bugs *before* any registered run
  and refused to patch either in passing (§4).

**The converse, which is where the real hole is.** The ladder-and-apple sentence
still has no passing claim. So do touch, tool use, shelter, sleep, plasticity,
proprioception, social, voice, smell, balance, thermal-kills, death-and-retry,
hunger/thirst, and fast/slow — **14 of 23**.

## 4. Is the builder alive and productive? — alive, honest, and slow

24 scheduled iterations in the last 24 h:

| outcome | count |
|---|---|
| blocked by the 90% stop (pre-reset, 18:07 Sun → 04:07 Mon) | 11 |
| lost to the 5-hour session limit on all three models (12:07/13:07/14:07) | 3 |
| **skipped by the new pace gate (18:07)** | 1 |
| ran, `rc=0` | 9 |

**PASS delta over 24 h: 83 → 84 (+1** — NE.00 at 06:29). Registry grew
**169 → 181 (+12)**. The demonstrated fraction therefore *fell*, 49.1% → 46.4%.

No repeated identical failures, no unresumed pause, no iteration aborting on
load. Liveness is genuinely proven, not claimed — the 16:07 iteration checked the
*worker's* CPU (99.6%), not merely that a pid existed. Every FAIL/VOID this week
was recorded with a clean stamp and a routing, and both re-runs were launched to
**re-certify**, not to re-roll. This is a builder behaving well under a tight
meter.

**RANK 2 — the Review organ is down and holds an unregistered backlog.**
`review.log`: `2026-08-24T06:37:03 review start` → `API Error: 529 Overloaded` →
`06:45:25 sweep end rc=1`. `review.sh` retries only on out-of-credits (line 44),
not on a transient API error, so one 529 costs the whole daily run. Meanwhile
three findings have been "routed to the weekly Review" and there is **no queue
file** — `grep` finds them only in a test docstring, `LOOP_JOURNAL.md`,
`LESSONS.md` and `CHAMPIONS.md`:

| routed | when | age |
|---|---|---|
| recipe sensitivity (lolr fixed A3, broke A4) — `8f6f750` | 2026-08-20 | 4 days |
| NE.01: graded vs knife-edged shelter occlusion — `e25d285` | today 16:15 | 2 h |
| **Water.apply pool-exit phantom force** — `d1bc3d1` | today 17:38 | 1 h |

**RANK 3 — the phantom-force staleness bill has a number, and nobody has written
it down.** `Water.apply` writes `xfrc_applied` only while a body is in the pool;
when the body leaves, the last force row stays applied forever. The decision *not*
to patch it in passing is correct and well argued — patching changes dynamics
under every certificate that ran with it. But the bill is currently a paragraph
in `LESSONS.md`, not a count attached to rows. **Five PASS certificates were
earned in a W0 carrying this bug: BA.01, LC.02, PS.02, PS.03, XL.00.** (Also
BA.02 VOID, LC.03 VOID, XL.01 FAIL, NE.01 FAIL — no claims at risk there.)

## 5. Compute honesty — the free quota is going to expire for the third week running

| week | Kaggle GPU-h charged | of 30 | expired unspent |
|---|---|---|---|
| 2026-W31 | 37.46 | — | — |
| 2026-W32 | 21.06 | 30 | 8.94 |
| 2026-W33 | 7.63 | 30 | **22.11** |
| **2026-W34** | **0.00** (no key in `gpu_budget.json`) | 30 | *in progress* |

Every hour actually charged produced a ledger row or a pre-registered diagnostic.
There is **no waste** in what was spent — this is entirely a non-spend finding.

**The pacing gate's causal premise does not survive the GPU record.** `lib_usage.sh`
argues the quota dies because "nobody was awake to press the button," and buys a
throughput cut to keep the loop awake through Sunday. But in W33 the loop was
awake Monday through Friday 12:07 and dispatched **7.63 h** — and W33's *first*
job was 2026-08-19 (Wednesday), so Monday and Tuesday dispatched **nothing at all
while fully awake**. The trend is 37.5 → 21.1 → 7.6 h. Awakeness is not the
binding constraint; an empty GPU-worthy queue is. Today the loop said so itself:
*"No GPU dispatched (nothing GPU-worthy; the NE family is CPU)."*

**And the gate is now costing iterations to protect that unspent resource.** At
18:07: `31% spent at 8% of the week (line 31%) — skipping`. By the gate's own
arithmetic (`allow = 25 + ⌈65·elapsed/100⌉`), the line stays at 31 until the week
reaches 10% elapsed, so **19:07, 20:07 and 21:07 will also skip** — the loop
resumes ~22:07 UTC. Steady state, an iteration costs ~1 point of `week:all models`
while the line rises ~0.39 points/hour: **roughly one iteration every 2.5 hours,
~10/day instead of 24.**

Two mechanical problems with the gate as written, independent of whether the
policy is right:

1. **`pace_gate` sits before `cd "$REPO"`** (`ladder_loop.sh:108`), so a skip
   fences out *zero-cost bookkeeping* as well as expensive work — including
   committing a finished run's ledger row (§6).
2. **It gates the wrong meter for the loop's actual binding constraint.**
   `_usage_pct` reads `week:all models` — 31% at 8% elapsed. But the builder runs
   on **Fable: 53% at 8% elapsed, 6.5× ahead of an even pace, and gated by
   nothing.** On that trajectory Fable exhausts around Tuesday, after which the
   loop falls back to Opus and draws *harder* on the very pool pacing protects.
   `lib_usage.sh` itself diagnoses the shared-pool problem and then applies the
   fix to the shared pool anyway.

## 6. Stuck decisions

`decisions --check`: **no `MEANS-ESCALATED`, no `OVERDUE`.** D1 (costs 38 specs)
and D10 (costs 8) are armed and due 2026-08-31. 10 undeclared, ratchet unchanged.

**Armed this audit: D7** (MovementMoodCoupling, open since 2026-08-13, evidence
complete for 11 days). Appended to `DECISIONS_NEEDED.md`; `class: goal`, default
= **option 3, accept-as-cosmetics on the record**, `decide_by: 2026-08-31`. That
default was chosen because it is the only one of the three that changes no code,
deletes nothing, and **narrows** what may be claimed (no spec may cite mood as a
behavioural channel) rather than widening it — a ratchet that shrinks. It is
explicitly *not* the loop's preferred option, which is 2.

**Nothing was quietly acted on.** The 26th audit's B1–B5 were all executed and
committed (`9449a1b`, `71c879f`). B6 was not — see RANK 1.

## 7. Bakeoff hygiene — no findings

`DECISIONS_RESOLVED.md`: 3 entries. PS.01/J is recorded as a VOID and was **not**
treated as a verdict — PS.01/J2 re-ran it and named a winner. D2 was resolved by
ledger replay with exposure 0 vs 9 and benefit 0, a learning gate, a named loser,
and a re-open trigger keyed to the quantity it rests on. No winner sits inside a
noise margin. LC.03 v2 concluded **VOID** ("fewer than two learners") and the
builder correctly refused to seat `wm-latent`, refused a third re-screen, and
sent the fork to the owner as D10 — a VOID treated as a VOID, which is the whole
point of §7.

---

## 8. The honest summary

**Marginally closer, and by subtraction rather than addition.**

The arithmetic is unflattering: in 24 hours the ladder gained **12 claims and 1
demonstration**, and the demonstrated fraction fell from 49.1% to 46.4%. Fourteen
of twenty-three constitutional commitments still have zero credited claim. The
ladder-and-apple sentence — *the* north star — has no passing spec and its whole
family sits behind D1, now sixteen days open.

But the day's two most valuable outputs were both **demolitions of things the
system believed**, and that is worth more than a green tick. DP.00's PASS turned
out to certify a 12×12 gridworld while being read aloud as "Jack's world rewards
deliberation" — the precondition of the entire fast/slow axis had never been
tested in the world he lives in. And DP.05's replay probes found two real W0
bugs, one of them a physics fault that reads as seed noise forever unless
something insists on byte-exactness, and the builder routed the fix rather than
patching it under five live certificates.

So the answer to "are we closer to a curious humanoid, or only to a longer list
of green ticks?" is: **neither, today.** We are closer to a ladder that is
*honest*, which is the necessary precondition for the first. What is being spent
to get there is time and free GPU quota — 30.9 hours expired in two weeks, a
third week starting at zero — and the mechanism the loop built to stop that loss
is currently making the loop dark instead of making it dispatch.

---

## FOR THE BUILDER

Ranked. B1 and B2 are the ones that matter; neither owes a re-run.

**B1 — Harvest DP.05 and obey its pre-registered fate. Do not add seeds.**
The registered run finished at **2026-08-24T18:30:15** (3,173 s, clean stamp,
commit `eacafe2`, `impl_sha a3facc5e6f8ceaa8`) and the row is sitting
**uncommitted** in `experiments/ledger.json`. It is `FAIL`, "pre-registered
threshold not met". The direction is right and the failure is *consistency*:

```
plan_h4   141.13 s ±16.47,  1.67 eats     react_k5   119.8 s ±0.22,  0.0 eats
plan_h10  133.17 s ±18.90,  1.00 eats     react_k10  119.8 s ±0.08,  0.0 eats
gap_s      13.27 s ±18.83   gap_clear 0.333 (1 of 3 seeds cleared MIN_GAP_S)
ref_span  173.1 s / 4.0 eats  (instrument alive)   probe_mismatch 0.0
ctrl_gain -0.0136 (< CTRL_TOL 0.02)   ctrl_gain_broken 0.112 (> floor 0.04)
```

Every VOID gate is clean, the control fired in the right direction, and the
instrument gate proved food pays. `_check` requires `gap_clear == 1.0` and a
3σ margin on `plan_h10` vs `react_best` (actual: 0.70σ). **This is a real FAIL
and it is informative**: planning eats and reacting never does, but a 21 s
lifespan edge does not survive a per-seed gate at this envelope. Commit the row,
record the FAIL and the routing in the docstring as you did for NE.01, and take
any envelope question to a *registered* redesign — not a wider seed set on the
same spec.

**B2 — Give "routed to Review" a file, and make one 529 not cost a day.**
Two small pieces, both durable:

- Create `docs/REVIEW_QUEUE.md` (or a `ROUTED:` marker in the same idiom as
  `COVERS:` / `DECIDE:`) with one row per routed finding: id, date routed,
  source commit, one-line question, and the **staleness bill** — the ledger rows
  that would be invalidated by acting on it. Seed it with the three open today
  (recipe sensitivity 08-20; NE.01 occlusion; Water.apply phantom force) and give
  Water.apply its number: **BA.01, LC.02, PS.02, PS.03, XL.00 — 5 PASS
  certificates**. Then a `--check` can print "3 routed, 0 acted on, oldest 4
  days" and the backlog stops being invisible.
- `scripts/review.sh:44` retries only on out-of-credits. Extend the retry to a
  transient API error (529/overloaded) the same way. Today's daily Review died on
  one and never ran.

**B3 — Fix the pace gate's two mechanical faults; leave the policy to the owner
(FOR THE OWNER §2).**

- Move `pace_gate` **after** the harvest/commit step, or add an explicit
  pre-gate bookkeeping path. Right now `pace_gate say || exit 0` sits at
  `ladder_loop.sh:108`, before `cd "$REPO"`, so a skip also blocks committing a
  finished detached run's ledger row — which costs no meter at all and is exactly
  what is stalled tonight.
- Log **both** meters in the PACING line, and name the one you act on. Today:
  `week:all models 31%` (gated) vs `week:Fable 53%` (ungated, 6.5× ahead of an
  even pace, and the builder's own model). The line as written reads as if the
  loop is 31% spent; on its own meter it is 53%.

**B4 — DP.05 is registered `COVERS: fast/slow (fixture)`, so it could never have
moved that commitment off zero.** `coverage.py` credits `claim` only. `eacafe2`'s
message justifies the unit as "Zero-pass commitment fast/slow (8 declared, 0
passing), per the standing rule" — true of the commitment, but not something
DP.05 can change by construction; a PASS would have printed as *"support passing,
not credited"*. The **work was still the right pick** (it unblocks DP.01–DP.03
and it caught the DP.00 world error). Either re-kind it in the registry with a
stated reason, or, when invoking the zero-pass rule, name the `claim`-kind spec
the unit is actually clearing a path to. One line either way; no threshold moves.

**B5 — carry-forward, third audit running.** UB.9 is still the **only**
claim-kind PASS behind both `hearing` and the 22-spec `one brain / unison`
family, and 71c879f correctly moved its conditionality into the registry. It now
needs the measurement that conditionality names — a per-arm must-learn target or
a recorded per-arm loss descent. Still not this week. Still not prose forever.

---

## FOR THE OWNER

Three items. §2 is the new one and it is the only one that costs you anything.

**1. D1 is sixteen days old and it is the reason the north star has no spec.**
Unchanged from the 26th audit and I am not re-arguing it — the evidence table is
complete. The cost, restated once because it is the whole point of this project:
*"If there is a ladder with an apple on top, he must try to climb the ladder,
fall, and learn from falling, purely out of curiosity"* (GOAL.md:30-32) has
**zero passing specs**, and CU.1–CU.7 — the family that would give it one — are
all behind D1. Curiosity's only credited PASS today is T2.08, a Tier 2 coverage
test. **If D1 is going to stay open, saying so is itself useful.**

**D7 has now been armed** (see below) and **D9** still has its bakeoff table
attached and needs no further number.

**2. NEW — the pacing fix went the opposite way from what was filed, and W34's
30 free GPU-hours are at zero on a Monday evening.**

I filed B6 last audit: when the Claude meter runs hot, spend one lean iteration
**dispatching** detached Kaggle work, because kernels compute through a blackout
for free and write their own receipts. What was built instead is a gate that
**skips iterations** to hold Claude budget for later in the week.

The evidence says that trade is priced against the wrong cause:

- W33: the loop was awake Monday–Friday and dispatched **7.63** of 30 h. Its
  first job was **Wednesday** — Monday and Tuesday dispatched nothing while fully
  awake. The trend is 37.5 → 21.1 → 7.6 h.
- W34 so far: **0.00 h**, no entry in `gpu_budget.json`, last job 3 days 10 hours
  ago, and the loop is now pace-gated dark until ~22:07 UTC.
- Cost of the gate at steady state: **~10 iterations/day instead of 24.**

So the loop is spending throughput to protect a resource it has demonstrated it
cannot consume — and the reason it cannot consume it is an empty GPU-worthy
queue, not sleep. Two things would help, and only the second is yours:

- *(builder, filed as B3)* fix the gate's mechanics — it currently also blocks
  free bookkeeping, and it reports `all models 31%` while the builder's own
  meter, Fable, is at **53% at 8% of the week and gated by nothing**.
- *(yours)* the **dispatch-then-idle carve-out** I asked for last audit is still
  the cheap fix and still needs one sentence: *when `week:all models` crosses
  ~80%, the loop may spend one lean iteration dispatching detached Kaggle work
  before it freezes.* It relaxes no limit. If you would rather the answer be "no",
  that is a fine answer and it closes the item — but then the honest conclusion is
  that ~20 free GPU-hours a week are not recoverable and the ladder should stop
  being sized as though they are.

**3. D7 is armed and will fire on 2026-08-31 if you do not rule.**

D7 (MovementMoodCoupling failed its ablation — T3.07, 3 seeds, 0.225/0.275/0.375
against chance 0.25) has been open since 2026-08-13 with complete evidence and no
deadline, which means silence deadlocked it forever. Under the standing rule I
have armed it. The pre-registered default is **option 3 — accept it as cosmetics,
on the record**: keep the module for companion UI, and **no spec may thereafter
cite mood as a behavioural channel**; GOAL's interoception claims must route
elsewhere. I chose that branch because it is the only one that deletes nothing,
writes no new model code, and *narrows* what may be claimed rather than widening
it. It is **not** the loop's preferred option — the loop's read is option 2 (route
mood in as an input token). Reversing it later costs nothing but your word.

---

*Instruments run this audit: `experiments.coverage` (exit 0),
`experiments.decisions --check` (0 means-escalated, 0 overdue, 10 undeclared),
`experiments.champions --check` (12 violations, ratchet 8/8 unchanged),
`experiments.run status`, `experiments.run verify` (0 failures on all 5 probes),
a per-PASS implementation / `git cat-file` commit-liveness / declared-control
cross-check over all 84 PASS entries, `git diff -U0 "@{7 days ago}" HEAD` over
`registry.py` / `registry_expansion.py` / `tests/` with every deleted
threshold-shaped line inspected, a `declarations()` × ledger join to compute
credited claim-kind passes per commitment, `gpu_budget.json` per-week
reconciliation, `scripts/lib_usage.sh` / `ladder_loop.sh` gate-order and
pace-line arithmetic, a W0-importer × ledger join for the phantom-force staleness
bill, and `/data/jack-logs/{ladder,overseer,review}.log` cadence counts.*
