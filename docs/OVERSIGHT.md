# OVERSIGHT — 61st audit, 2026-09-02 06:40 UTC (HEAD `d36f3f9`, 0 unpushed, tree DIRTY: one runner-written `experiments/ledger.json` row awaiting harvest)

## VERDICT: INTEGRITY RISK — `T3.09` recorded a **FAIL** at 06:33 whose own wrong-goal control cleared the exact margin the claim needed (`shuf_gain 12.47 ≥ MARGIN_AFF 11.0`), and that row arms a `kills` clause that deletes 559 lines of code

The counterweight first, because it is large and it is real.

**Section 1 is mechanically clean.** All **94 PASS rows** have an importable
implementation on disk, a `commit` that still resolves in git, and — for every
row whose spec declares a control — a `control_fn` actually wired into
`run_spec`. The only two PASS rows without `control_metrics` (`T0.01`, `T0.10`)
declare `control="NONE, BY DECISION"` on their face with the 52nd audit's
reasoning attached. Zero findings.

**Section 2 is completely clean.** Across `experiments/` and `scripts/` since
the 60th audit closed at 00:45, exactly **three** already-existing named
constants changed value, and **all three moved in the tightening direction**:
`N_PROPERTIES 12 → 13` (T0.31), `TEMP 0.25 → 1.0` (LG.10 v2 — more sampler
entropy makes match/unanimity/swap_agree/null all strictly harder), and
`N_LIVES 16 → 32` (T3.09 v2 — sample size, with every gate constant
`MARGIN_AFF 11.0 / MIN_AFFECTED 8 / OFF_MIN_FED 0.5 / CORE_SHRINK 0.5`
untouched and the commit made *before* the run). No control was deleted or
weakened, no `_check` gained an `or`, no seed count was reduced, no assertion
was removed. The builder also committed both repairs before running them, and
in both cases the repaired run bought a **RED**, not a green — LG.10 FAIL and
T3.09 FAIL. That is the discipline working.

**And the 60th audit's own FINDING 1 was executed.** The backlog reader was
blind to 6 of 26 rows; it now reads **28 of 28**. `D1.0`'s 16.17 VOID GPU-hours
now have an owner and a clock (`d10-successor-rerun-under-adopted-gate`,
DUE 09-06). Both repairs verified live, not taken on the commit message's word.

Findings are ranked by damage to the trustworthiness of the ledger.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | 0 commitments with NO spec. **4 CLAIM-DEAD** (smell, balance, shelter/building, thermal). **12 of 23 commitments have ZERO passing claim spec.** Standing red by design until the 09-06 Review window. |
| `decisions --check` | 0 | 3 armed (`D15` 09-05, `D16` 09-05, `D17` 09-07). 0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE. Ratchet **0/10 undeclared**. |
| `champions --check` | 0 | 27 seats. 1 ARENA-UNREACHABLE (Fast/slow coupling, rooted at `LC.03`), 3 NO-ARENA (ASR, Speaker ID, Language grounding), 1 UNCONTESTED (Vision encoder). Ratchet ok — **but see FINDING 5**. |
| `run review-queue` | 0 | **24 OPEN, 2 HELD, 2 ACTED of 28 routed**; oldest live 9 d; consumer ran 1 d ago. 0 violations — **but see FINDING 3**. |
| `review_liveness` (schedule half) | 0 | Silent. The desk kept its schedule. |

**On "arm at least one decision per audit":** vacuous today, and stated rather
than faked. `decisions.py` reports 0 of 10 undeclared — every open entry
already carries a class, a default and a `decide_by`. There is nothing to arm.

---

## FINDING 1 — A recorded verdict whose own control impeached the rig, and it is armed to delete 559 lines (RANK 1)

`experiments/ledger.json`, `T3.09` attempt 3, `commit d36f3f9`, ran
`2026-09-02T06:33:18`, **uncommitted in the working tree at audit time**:

```
status                   FAIL      ("pre-registered threshold not met")
creative_contribution    -9.96     vs MARGIN_AFF 11.0        → the claim lost
shuf_gain                +12.47    vs MARGIN_AFF 11.0        → THE CONTROL WON
seeds                    [0]       n_affected 11
```

`shuf_gain = off_ttf_aff 146.65 − shuf_ttf_aff 134.18 = 12.47`. The
**wrong-goal control** — the identical trained loop handed a goal reflected
through the rover, "exactly as much information, exactly wrong" — **cleared the
margin the claim itself had to clear, and the claim did not.**

The spec's own control text in `registry_expansion.py:1645` is unconditional:

> *"if wrong-goal advice helps as much, the site rewards any detour
> perturbation and **the test measures nothing**."*

So is `SYSTEM.md` law 2 — *"A control that also passes means the test measures
nothing"* — which sits in **class 3, CONDUCT: fixed, and not up for
measurement.**

**Why it recorded FAIL anyway.** `experiments/tests/t3_09_creative_loop.py:489`:

```python
    if m["creative_contribution"] < MARGIN_AFF:
        return False                # ← returns here
    # ── a PASS must survive its control (law 2) ──
    if c["shuf_gain"] >= MARGIN_AFF:
        return Status.VOID          # ← unreachable whenever the claim fails
```

The control-vacuity lane is scoped **PASS-only**, matching the impl docstring
(line 92: *"a PASS whose shuf control ALSO cleared the margin"*). It was
pre-registered that way and the builder did not touch it after seeing the
number — that matters and I record it. But a spec's docstring may not narrow a
class-3 law, and the narrowing is load-bearing here: the one run in which the
lane would have fired is the one in which it cannot be reached.

**Independent corroboration that this rig carries no signal.** The four arms
rank *anti-correlated with advice quality*:

| arm | advice | mean ttf on affected lives |
|---|---|---|
| `shuf` | goal reflected — **deliberately wrong** | **134.2 s** (fastest) |
| `off` | none (shipped random detour) | 146.7 s |
| `loop` | the claim | 156.6 s |
| `twin` | 3-line goal subtraction — **correct** | **191.2 s** (slowest, −44.5 s vs no advice) |

Wrong advice helps by 12.5 s; correct directional advice hurts by 44.5 s. At
**one seed and n = 11 affected lives, with no variance reported.** `T3.09`
declares no `seeds=`, so it inherits `seeds=1` — against GOAL.md's *"at ≥3
seeds where the claim is about learning"*, for a spec that trains a proposer
(`train_pairs 32`, `train_accept_rate 0.06`).

**Why this is RANK 1 and why no instrument caught it.** Every honesty
instrument in this repo is built to catch a violation that buys a **green**.
This one buys a **red** — and the red is not inert. `T3.09`'s spec carries
`kills="AlphaGeometryLoop.py (559 lines) — wire it or delete it"`, and the impl
docstring at line 95 says in its own words: *"WHAT A FAIL EXECUTES … delete
`AlphaGeometryLoop.py` from the root."* The next iteration harvests this row.
A 559-line deletion is about to be executed on a run whose control cleared the
claim's margin.

I am not asking for the number to be re-bought or the verdict softened. The
honest repair is in **FOR THE BUILDER B1** and it makes this row *weaker*
(FAIL → VOID under the corrected ordering), which costs the ladder a
demonstrated point. That is the direction a real repair goes.

---

## FINDING 2 — The spec that decides whether the *creative* loop earns its existence measured **zero** creative-branch invocations (RANK 2)

Across all three T3.09 attempts, on every arm:

| attempt | `loop_consults` | `loop_direct` | `loop_creative` | `loop_none` | `shuf_creative` |
|---|---|---|---|---|---|
| 2 (`db4e0e7`) | 48 | 6 | **0** | 42 | 0 |
| 3 (`d36f3f9`) | 77 | 33 | **0** | 44 | 0 |

**142 consults across two recorded runs and the creative branch fired not
once.** `solve` answered from its direct branch or returned `None`, always.

The impl pre-registers this as reportable and untuned (line 111), so it is
honestly disclosed, not hidden. But it is material to the action the row
arms: a FAIL here cannot distinguish *"the creative machinery does not help"*
from *"the creative machinery never ran"*. Deleting 559 lines named
`AlphaGeometryLoop` on evidence in which the AlphaGeometry loop never took its
creative path is a deletion whose evidence does not name the thing deleted.

Combined with FINDING 1, the row supports exactly one honest sentence: *at this
call site, at n=11 on one seed, detour advice of any kind is noise.* That is a
finding about **the site**, not about the module — the same shape as `DP.00`'s
"the finding is about the world, not about him".

---

## FINDING 3 — **18 of 28** review-queue rows now carry `DUE: 2026-09-06`. One Sunday holds eighteen dated promises (RANK 3)

```
DUE: 2026-09-05   1 row
DUE: 2026-09-06  18 rows
```

The builder's own 01:19 journal flagged this at **eleven**. Four hours of
honest routing took it to **eighteen** — every new row landing on the same
Sunday because 09-06 is the next FULL run.

Today this is **not a violation** (`review-queue` rc=0) and I am not calling it
one. It is a forecast, and it is the one the tool's own doctrine says is worst:
if the FULL run cannot clear eighteen designs in one sitting, **seventeen-plus
rows go OVERDUE simultaneously on 09-07** — the queue's strongest signal, a
dated promise broken in the open, fired en masse. And the tempting response
(re-arm all eighteen with a new date) is precisely the deadline-that-moves-when-
it-is-reached failure the `DUE:` mechanism replaced.

Two of the eighteen (`ne01-occlusion-knife-edge`, `water-apply-phantom-force`)
are HELD behind `w0-too-shallow`, which is itself DUE 09-06 — so a slip on one
row cascades to three.

The honest repairs are available **now, before the clock runs**: stagger the
clocks with stated reasons, or DECLINE the rows that will not be taken. Both
are ratchet-legal; a mass re-arm on 09-07 is not.

---

## FINDING 4 — Detached spec runs are still undeclared to procwatch (third occurrence) (RANK 4)

```
2026-08-31T19:12:41  LEFTOVER PROCESS  — experiments.decisions --check
2026-09-01T04:27:26  LEFTOVER PROCESS  — python -            (146s CPU)
2026-09-02T05:39:54  LEFTOVER PROCESS  — python -c … _module_for … (178s CPU)
```

The 05:39 leftover was **T3.09's own attempt-2 run**, launched as an inline
`python -c` rather than through `dispatch.sh` / `launch_detached.sh`, the two
paths that declare what they detach. The watcher does the right thing — names
it, does not kill it — but the loop closes each of these slots with
`LEFTOVER=1` and the process is legitimate. Low damage, but it is the third
time, and a watcher that cries wolf on its own runner is a watcher that will be
ignored on the day the process is not legitimate.

---

## FINDING 5 — `champions.py` reports the Learning core seat "ok" while it is held **BY VERDICT** off a run that VOIDed (RANK 5, corroborating, already routed)

```
Learning core    decl  BY VERDICT   ok
                 arena: LC.00 LC.01 LC.02 LC.03 LC.04 LC.05 LC.06 LC.07
```

`LC.03` is VOID-FORECLOSED. `D10`'s armed default fired on 09-01 and seated the
core BY VERDICT — the file's strongest marking — off a VOID row. `champions.py`
prints `ok` because its ratchet asks only whether the arena **resolves**, never
whether the verdict behind the marking was a verdict at all.

This is audit item 7's canonical failure (*"a VOID treated as a verdict"*) and
it is **already routed** — three queue rows carry it
(`d10-learning-gate-uses-two-different-denominators`,
`d10-learning-gate-sits-at-the-untrained-twin-level`,
`d10-successor-rerun-under-adopted-gate`, all DUE 09-06). I restate it only to
record the *instrument* gap: no tool in this repo can read the strength of the
evidence behind a `HELD:` marking, so the strongest marking in `CHAMPIONS.md`
is the one with the least verification behind it.

---

## SECTIONS WITH NO FINDINGS (stated plainly, because that is a result)

**3 — Drift from the goal: none.** Every unit in the last 24 h traces to a
GOAL.md sentence:

| unit | GOAL.md sentence |
|---|---|
| `LG.02` PASS (3 seeds) | *"his diary records whose advice proved true, so trust in a person can be earned and checked"* (line 144) |
| `LG.10` FAIL ×2 | *"The LLM is his mouth, never his mind"* (line 159) |
| `T3.09` VOID×2 + FAIL | Tier 3, *"every component ablated; dead weight deleted"* (line 268) |
| `UB.10` VOID harvest | *"Unison — senses fused"* (line 269) |
| `D1.0` / `LC.07` pilot | ARCHITECTURE, always contested (`SYSTEM.md` class 2) |
| `T0.12/17/21/27/31` re-buys, audit items | *"protects the honesty of watching what happens when the three meet"* (line 8) |

**4 — Builder alive and productive.** 28 iteration starts, 29 ends in the 24 h
to now; **every one rc=0**. PASS delta **93 → 94** (`LG.02`). Two additional
honest REDs bought (`LG.10` FAIL, `T3.09` FAIL). No paused loop, no repeated
identical failure, no load abort, no credit exhaustion.

**5 — Compute honesty: clean.** Week 2026-W35: **19.20 GPU-hours** across 12
jobs (18.93 Kaggle + 0.27 Colab), **0.0 failed hours** — the cleanest week in
the file (W32 carried 1.18 failed hours). ~10.8 h of the 30 h Kaggle quota
remains with 4.5 days to Sunday's reset. Every W35 job traces to a ledger row
or a recorded pilot; the one previously-orphaned spend (`D1.0`, 16.17 h → VOID)
was given an owner and a clock by the 60th audit's B2 and is now
`d10-successor-rerun-under-adopted-gate`.

**6 — Stuck decisions: nothing actionable.** 3 armed, 0 MEANS-ESCALATED, 0
UNDECLARED, 0 OVERDUE. No owner decision was quietly acted on: the eleven
defaults that fired on 09-01 are each recorded in `DECISIONS_RESOLVED.md` with
losers, evidence and a reversal path.

**7 — Bakeoff hygiene:** one live issue, and it is FINDING 5 above (already
routed). No winner chosen inside a noise margin in the window; no decision made
without a learning gate.

---

## 8 — THE HONEST SUMMARY: are we closer to a curious humanoid that climbs the ladder?

**Yes, by one real rung — and it is a rung, not a tick.**

`LG.02` is the owner's own liar test, queued since 2026-08-09, and it passed at
**three seeds** with a live null (0.028 divergence with attribution stripped,
join alive), a swap control that migrated (0.711 / 0.733), a prior measured at
exactly 0.5 for both voices, and 0.95 attribution recall. Worst-seed
last-quarter divergence 0.689 ± 0.103 against a 0.40 gate. That is *"one
sentence can spare him a thousand falls, and his diary records whose advice
proved true"* turned into something that could have failed and did not. It is
the single most GOAL-shaped result in weeks.

**And the honest counterweight, which is heavier.** Of the owner's 23
constitutional commitments, **12 have zero passing claim spec** — touch, tool
use, proprioception, sleep, hunger/thirst, fast/slow, plasticity, death &
retry, plus the four now formally **CLAIM-DEAD**: smell, balance,
shelter/building, and *too cold kills him*. `one brain / unison` carries 23
specs and **one** pass. 85 of 217 specs (39%) are unreachable. The board reads
94/217 and four of the owner's own sentences have nothing runnable behind them
at all.

What today's audit adds to that picture is narrower and sharper. The system is
now good enough at buying honest REDs that it has produced a new failure mode
its instruments were never built for: **a violation that costs the ladder a
point instead of buying one.** `T3.09` is about to delete 559 lines on a run
whose deliberately-wrong control beat every other arm, in which the branch
named in the spec's title never once executed, at one seed. Nothing in
`coverage`, `decisions`, `champions` or `review-queue` can see that, because
all four were built to catch optimism.

So: closer, honestly, by one rung — and one action away from recording a
deletion the evidence does not support.

---

## FOR THE BUILDER

**B1 (do this first, before harvesting `T3.09`). Do NOT execute the `kills`
clause on the 06:33 row. `AlphaGeometryLoop.py` stays for now.**
The row itself is honestly recorded and must be committed as the runner wrote
it — do not edit it, do not re-run for a better number. What must not happen is
the *action* it arms.
Then do all three of:
  1. **Route it.** Add a `REVIEW_QUEUE.md` row `t309-control-clears-the-claims-
     own-margin` with a `DUE:` that is **not** 2026-09-06 (see B2), carrying the
     four numbers: `creative_contribution -9.96`, `shuf_gain +12.47`,
     `MARGIN_AFF 11.0`, `loop_creative 0` on 142 consults, `seeds [0]`,
     `n_affected 11`.
  2. **Fix the lane ordering, pre-registered, before any further T3.09 run.**
     In `_check`, the control-vacuity test belongs with the other rig gates —
     *above* the claim branch, not below it:
     ```python
     if c["shuf_gain"] >= MARGIN_AFF:
         return Status.VOID   # site rewards any perturbation — law 2, unconditional
     if m["creative_contribution"] < MARGIN_AFF:
         return False
     ```
     Amend the impl docstring's VOID list to drop the words *"a PASS whose"* —
     `SYSTEM.md` law 2 is class-3 CONDUCT and is unconditional; a spec docstring
     may not narrow it. **Be explicit in the commit message that this repair
     converts the recorded row from FAIL to VOID and therefore costs the ladder
     a point** — that is why it is not gate-fitting.
  3. **Raise the seed count before the next attempt.** `T3.09` inherits
     `seeds=1`. A verdict that executes a 559-line deletion, on a metric whose
     four arms rank anti-correlated with advice quality, needs `seeds=3` at
     minimum. Declare it in the registry in the same commit as (2), before any
     run.

**B2. Break up the 09-06 pile-up now, while it is still a forecast.**
18 of 28 rows are DUE 2026-09-06. Before Sunday, either stagger the clocks with
a stated reason per row, or DECLINE the rows that will not be taken. Do not
wait for 09-07 and re-arm eighteen at once — the queue's own doctrine names
that as the failure the mechanism replaced. If the Review genuinely can clear
eighteen designs in one sitting, say so in the file and leave them; the point
is that this be a decision recorded in advance, not a discovery made after the
dates go red.

**B3. Declare T3.09's runner launch to procwatch.**
Third `LEFTOVER=1` in three days, and the 05:39 one was T3.09's own detached
run. Route inline `python -c` spec runs through `launch_detached.sh` (or add
the declaration hook the other two paths already use) so the watcher stops
flagging the loop's own legitimate work.

**B4. Give `champions.py` a way to see a VOID behind a `HELD:` marking.**
The Learning core seat prints `ok` while held **BY VERDICT** off `LC.03`, which
is VOID-FORECLOSED. The tool's docstring already proposes per-seat `HELD:` /
`ARENA:` markers; the addition this audit asks for is one conjunct on top of
that: when a seat is marked BY VERDICT, resolve the deciding spec's ledger row
and report `VERDICT-IS-A-VOID` if its status is not PASS or FAIL. Count it in
the ratchet. This is the instrument gap FINDING 5 names; the `D10` rows already
handle the specific case.

---

## FOR THE OWNER

**Nothing new is escalated to you.** All three open decisions (`D15`, `D16`
09-05; `D17` 09-07) are armed with defaults and need no action to remain safe.

Two things are worth your eye, neither blocking:

1. **A 559-line deletion was one iteration away from executing on evidence its
   own control impeached** (FINDING 1). The system caught it here rather than
   in a post-mortem, and B1 stops it — but note the shape: every honesty
   instrument you have was built to catch a violation that buys a *green*. This
   one bought a *red*, and none of the four saw it. If you want that gap closed
   structurally rather than by an overseer reading a diff, that is a decision
   for you, and I have not opened one on your behalf.

2. **Four of your own sentences are now formally CLAIM-DEAD** — smell, balance,
   shelter-building, and *too cold kills him* — meaning every claim spec behind
   them is parked or foreclosed, with a redesign routed and nothing runnable.
   Twelve of twenty-three commitments have no passing claim at all. The 09-06
   Review is where those redesigns are scheduled to be written, which is
   exactly why FINDING 3's eighteen-row pile-up on that one Sunday matters more
   than its rc=0 suggests.

The good news is real and is yours: **your liar test passed.** `LG.02`, queued
2026-08-09, three seeds, live null, migrating swap control — the liar loses him.
