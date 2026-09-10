# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Running history lives in git.

**2026-09-10 06:37–07:0x UTC — 87th audit.** Window: the last 24 hours
(2026-09-09 06:50 → 2026-09-10 06:49).

## VERDICT: DRIFTING

**The ledger is sound, nothing was loosened, and no instrument is lying.** The
finding is that the builder has now been dark for **46.7 hours**, and — this is
the new part — **`pace_gate` will not release it before the week resets itself
on 2026-09-14, and roughly a third of the reason is the spend of the organs
auditing the outage.** Including this one.

---

## RANK 1 — the blackout is not a wait. It is an equilibrium, and we are in it.

The Review reached the same headline independently at 06:44 this morning
(`6e79f7d`, `D26` evidence addendum) and its numbers are correct; I re-derived
every one of them rather than quoting. **Three things it did not compute follow
below, and all three make the picture worse.**

### The measured state

| | |
|---|---|
| builder last iteration | `2026-09-08T08:23` — **46.7 h ago** |
| consecutive `PACING:` skips | **46 of 46 slots** |
| builder commits in 24 h | **0** |
| ledger settlements since the blackout began | **0** |
| demonstrated | **108 / 245** — unmoved since `2026-09-07T11:25` (**67 h**) |
| `week:all models` | 68% — against a **90%** hard stop nobody is near |

### (a) The closing rate is not 2.3 points/day. It is 0.29, and we spend the rest.

The Review computed the pace line (9.3 pts/day) against the **external** draw
(7.0 pts/day) and concluded the gap closes at 2.3/day — ~6 days. But the meter
also carries **our own** spend, and the honest reading is the 06:37-to-06:37
delta, which contains everything:

```
09-07 06:37   2%  ->  09-08 06:37  23%   = +21 pts/day   (builder awake)
09-08 06:37  23%  ->  09-09 06:37  59%   = +36 pts/day   (builder dark from 08:23)
09-09 06:37  59%  ->  09-10 06:37  68%   =  +9 pts/day   (builder dark all day)
                                   ^ +2 ours (review +1, overseer +2) + 7 external
```

    pace line rises                          9.29 pts/day
    meter rises, TOTAL measured               9.0  pts/day
    ------------------------------------------------------
    the line closes on the meter at          +0.29 pts/day

**The gap is 14 points (68 − 54). At 0.29/day that is 49 days.** The week resets
2026-09-14 05:23, 3.95 days away, by which time the line will have recovered
**1.1 of the 14 points it needs**. `pace_gate` does not release the builder this
week under any measured rate, and the margin is not close — it is off by 13×.

**The blackout therefore ends on 2026-09-14 05:23 by calendar rollover, not by
any decision: 2026-09-08T08:23 → 2026-09-14T05:23 = 141 h = 5.9 days.** That
exceeds the 4.3-day August blackout and is the largest loss of builder capacity
in this project's history.

### (b) Our own oversight apparatus consumes 87% of the headroom.

This is the part that indicts my own organ, so I am stating it first among the
consequences. Of the line's 2.29 pts/day surplus over the external draw, the
overseer and the Review spend **2.0** — leaving 0.29. Measured per-run cost this
week from `usage_ledger.jsonl`: review **1.00 pts/run**, overseer **0.67
pts/run**, builder **0.64 pts/run** over 28 runs.

Remove the two document organs and the line closes at 2.29/day and the builder
wakes in 6 days. Leave them in and it closes at 0.29/day and the builder never
wakes. **The organs auditing the outage are the marginal cause of its
continuation**, and the builder's own iterations were never the expensive
thing — 28 of them cost 18 points on 09-07, the day `demonstrated` went 106→108.

### (c) The gate is applied to exactly the wrong two organs.

```
scripts/ladder_loop.sh:183   pace_gate   <- builder, EVERY slot, no exemption
scripts/overseer.sh:79       pace_gate   <- overseer, 1 of 4 audits exempt daily
scripts/review.sh:30         usage_gate only   — NOT PACED
scripts/field_watch.sh:32    usage_gate only   — NOT PACED
```

`pace_gate` was built to stop the project going dark. It is applied to the one
organ that writes ledger rows, with no exemption, and not applied at all to the
two organs that write prose. The result is visible in the commit log without any
inference: **all 6 commits in the last 24 h are documents about the outage,
written by the organs the outage does not touch.** Section 8 of my brief asks
whether we are only accumulating green ticks; the answer this week is worse —
we are accumulating *commentary*, on the budget that would have bought evidence.

### (d) The release, if it came, would put the builder on the metered model.

`crontab` runs the loop `JACK_LOOP_MODEL=fable`. `week:Fable` reads **100%**
against `MODEL_FLOOR=95`, so `model_gate` refuses Fable and the chain falls to
Opus — which has no separate weekly line, fails open, and bills `week:all
models`, **the meter `pace_gate` is rationing**. The gate would release the
builder onto the only model whose spend immediately re-crosses the line it just
cleared. At 0.64 pts/iteration the builder would re-dark within two slots.

### What this costs, in the currency the gate exists to protect

`experiments/gpu_budget.json` has **no `2026-W37` key at all**: 0.00 of 30 free
Kaggle GPU-hours, expiring Sunday 2026-09-13 — before any projected wake.
Prior weeks: W32 16.61, W33 7.89, W34 1.62, W35 19.20, W36 17.73. `pace_gate`'s
own justifying comment states its purpose as *"the loop is still awake when the
GPU quota expires."* It is now the reason the loop is not.

---

## RANK 2 — three armed defaults are queued behind an organ that is switched off

The armed-default mechanism exists to break deadlocks caused by **owner
silence**. It has now deadlocked on **builder absence**, a case it has no clause
for.

| entry | state | its default requires |
|---|---|---|
| `D22` | OVERDUE since 2026-09-09T00:00 | firing (writes nothing) — **still unfired 2 days on** |
| `D18` | **OVERDUE as of 2026-09-10T00:00 — new today** | builder code (`lib_procwatch.sh`, `run_spec`) |
| `D26` | `decide_by` is **today**; red tomorrow | builder code (`pace_gate` attribution print) |

The 86th audit appended `D22`'s overdue notice and routed the firing to the
builder as its B1. The builder has not run since. Every armed default in this
project's history — `D1`, `D3`, `D4`, `D7`, `D8`, `D9`, `D11`, `D13`, `D14`,
`D15`, `D16`, `D17`, `D21` — is stamped *"fired … (builder)"*, and `D13`
records that the overseer may not edit its own script. So the routing is
correct and the queue is real.

**`D18`'s overdue notice is appended to `docs/DECISIONS_NEEDED.md` by this
audit, with the required wording, and its firing routed as B1 below** — the
`D22`/`D17` precedent exactly. I am not extending any deadline.

---

## RANK 3 — a finding I would have got wrong, corrected by a concurrent organ

At 06:40 `review-queue` reported **2 OVERDUE violations** (`d10-learning-gate-
uses-two-different-denominators`, `d10-learning-gate-sits-at-the-untrained-twin-
level`), both reading *"EXECUTION … owed by the BUILDER"*. I had them drafted as
builder debt dammed behind the blackout.

They were neither. The Review, running concurrently, disposed both at `d582acc`:
**the work landed 2026-09-06 at `8f2990d`, three days inside the clock — the
violations were a missing ACTED marker, not missing work.** Recorded because the
correction runs against my own thesis, and because a report written from a
06:40 snapshot would have overstated the builder's debt in the same audit that
argues the builder is being starved.

Queue as of this writing: **0 violations, EXIT 0.**

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN, no findings.** 143 rows: 108 PASS, 22
FAIL, 13 VOID. Checked every PASS independently:

- implementation exists in `experiments/tests/` — **108 / 108**
- `commit` still resolves in git (`cat-file -e`) — **108 / 108**
- spec declares a `control` — **108 / 108**
- non-empty `control_metrics` — **106 / 108**

The two exceptions are `T0.01` (repo imports clean) and `T0.10` (Kaggle job
round-trip). Both declare `control: "NONE, BY DECISION (52nd audit B5)"` with a
stated reason — an import either raises or it does not; a sabotaged upload fails
on the service's side. Declared, reasoned, not a silent gap. **No finding.**

**2. Thresholds and controls — CLEAN, no findings.** There are **no builder
commits in the window**, so nothing new to scan. Over the full 7 days the one
edit moving in the loosening direction is `MEASURED_DISCHARGE_CAPACITY 1 → 6`
(`cd5a27b`), and it is exemplary rather than suspect: it cites six discharging
commits by hash, calls 6 *"the demonstrated one-cycle MAXIMUM, not a sustained
rate"*, prints the before/after effect on its own amber count (6 flagged dates →
2), and explicitly refuses to let the constant carry the drain's truth — *"the
drain line, never this constant, carries that."* Justified by measurement.

*One honest caveat, not a finding:* today's *"13 rows share 2026-09-13 against a
measured capacity of 6"* is therefore measured against the best single day this
desk has ever had, while trailing throughput reads **0.43 disposed/cycle**.

**3. Drift from the goal.** The builder worked on **nothing** — it was switched
off. `demonstrated` last moved 2026-09-07T11:25 (107→108, `PL.00`). The last
three ledger writes before the blackout were `T0.29`/`T0.31` certificate
re-buys — the ladder certifying its own instruments. That serves GOAL.md's
*"protects the honesty of watching"* clause, which is real, but it is not brain,
body, or world.

The converse question, which matters more. From `coverage`: **4 commitments are
CLAIM-DEAD** (every claim spec parked or foreclosed) — **smell, balance,
shelter/building, thermal (kills)** — and **9 more have live claim specs with
nothing passing**: touch/contact, tool use, told world, proprioception,
plasticity, sleep, hunger/thirst, death & retry, fast/slow. The three GOAL.md
claims most at risk of quiet neglect read: **curiosity 2 of 12 passing**,
**one brain / unison 1 of 27**, learning-by-living gated behind `W0`/`W1`.
Zero commitments have no declared spec — the `coverage` `EXIT 2` is the
pre-existing, ratcheted CLAIM-DEAD and empty-class population, not a new hole.

**4. Is the builder alive and productive?** Alive, zero productive. 46 slots
fired, 46 refused, `rc=0` every time — the refusal and the heartbeat are written
by the same code path, which is the lesson the 86th audit already committed
(`0c367a2`). Covered in RANK 1.

**5. Compute honesty.** No GPU hours spent and none wasted — the waste is the
opposite failure. **W37: 0.00 of 30, expiring Sunday.** Additionally, `coverage`
reports `gpu<20min` among **3 cost classes that are NEWLY EMPTY with no path
in** — nothing runnable to implement and nothing gate-provisional to pilot — so
even an awake builder has nothing registered to dispatch there. Both levers are
the owner's; there is no third one the system can pull for itself.

**6. Stuck decisions — CLEAN apart from RANK 2.** `decisions --check` EXIT 0:
**0 `MEANS-ESCALATED`** (no fork a measurement could settle is sitting on the
owner's desk), **0 `UNDECLARED`** of 10 armed, 0 unrouted owner asks, 0 vanished
owner asks. Nothing to arm this audit — the ratchet is at its floor. No owner
decision was acted on without being recorded.

**7. Bakeoff hygiene — CLEAN, no findings.** No decision was resolved in the
window (the organ that fires them is off). Spot-checked `DECISIONS_RESOLVED.md`:
every armed-default resolution carries its firing date, its executing organ, and
an explicit statement of what the default did and did not change. No VOID
treated as a verdict; no winner chosen inside a noise margin.

**8. The honest summary — are we closer to a curious humanoid?**

No. And this week we are not even closer to a longer list of green ticks: the
list has not moved in 67 hours. What grew is the record of why it did not — six
commits in 24 hours, every one of them prose about the outage.

The specific thing worth saying plainly, because it is the shape this section
exists to catch: **the project's oversight apparatus is now healthy, fast,
well-instrumented, mutually corroborating — and is spending 87% of the budget
headroom that would have restarted the thing it oversees.** Four organs produced
an excellent diagnosis of a stalled builder this morning, at a cost that helps
keep it stalled. Every instrument is working. The creature has not moved since
Monday.

---

## FOR THE BUILDER

You will read this after a multi-day outage with a budget of roughly **one
iteration per three days** until the week resets on 09-14. The docket below is
over-subscribed against that. **Do B1 first and alone if you get one slot.**

**B1 — Fire two armed defaults. Cheapest possible acts; both are overdue.**
   - **`D22`** — routed by the 86th audit as its B1 and still unfired. Default
     **(i) THE RULE STANDS**; it writes nothing. Journal it with the required
     wording: *"the owner did not rule by 2026-09-08, so the pre-registered
     default fired"*, and record that it is reversible at any later date at no
     cost because it writes nothing.
   - **`D18`** — went red 2026-09-10T00:00; the overdue notice is appended to
     `DECISIONS_NEEDED.md` by this audit. Default **MEASURE AND REPORT, GATE
     NOTHING, RELAX NOTHING**: `lib_procwatch.sh` reads
     `/proc/PID/status:VmHWM` while walking pids it already resolves and NAMES
     any project python over the ceiling (name, never kill), and `run_spec`
     records `peak_rss_mb` from `resource.getrusage(RUSAGE_CHILDREN)`. The
     ~1.5 GB figure in `SYSTEM.md` **stands verbatim** — not raised, not
     narrowed, not annotated. Journal wording: *"the owner did not rule by
     2026-09-09, so the pre-registered default fired"*. Reversal: revert the
     two commits.

**B2 — Count the dark slot, and name the model.** The 86th audit's B2 and the
Review's `FOR THE BUILDER` item 1 both ask for this; add one field. On waking,
append to `docs/LOOP_JOURNAL.md`: consecutive slots skipped, the `week:all
models` reading that released you, **and which model actually ran**. The third
field is new and load-bearing: `week:Fable` is at 100% against `MODEL_FLOOR=95`,
so you will wake on **Opus**, billed to the shared meter, and nothing currently
records that substitution in a place a later audit can find.

**B3 — `D26` option (iv), only if `D26` has been ruled or its default has
fired.** Do not pre-empt it. As specified by the Review: `pace_gate`'s skip line
additionally prints our own attributed spend beside the shared total, summed
from `usage_ledger.jsonl`'s existing start/end pairs, and consecutive dark slots
become a ratcheted metric. **Gate nothing, change no behaviour.** No spec
declares `lib_usage.sh` in `IMPL_DEPS`, so no certificate is staled.

**B4 — New, from this audit's (b): the pacing arithmetic has no reader.** No
instrument in this repo can print *"the line closes at +0.29 points/day; at this
rate the builder wakes in 49 days; the week resets in 4"*. Both the Review and I
computed it by hand this morning, from a shell, twice, and got different answers
because we included different terms. Add it to the existing pacing print as
**measurement only, gating nothing**: line slope, total measured meter rise over
the trailing 24 h, the resulting closing rate, and the projected release date
beside the week-reset date. **State the terms it includes.** The defect this
catches is not the outage — it is that a 5.9-day foreclosure was legible only to
whoever happened to do the arithmetic by hand, and the organ it forecloses is
the one that cannot.

---

## FOR THE OWNER

**1. There is a one-line lever, it is already permitted, and it is yours.**

`pace_gate` returns 0 — proceed — on the mere existence of `.usage-resumed`
(`lib_usage.sh:76`), which the file's own comment calls the owner override:
*"an explicit 'make it continue' outranks a smoothing heuristic."*

    cd /home/opc/jackthelearner
    printf 'ceiling=90\nuntil=%s\n' "$(date -d '2026-09-14 05:23' +%s)" > .usage-resumed

This suspends **pacing only**. The 90% hard stop (`usage_gate`) stays fully
armed and is what the `ceiling=` line feeds; at today's 68% there are 22 points
of headroom beneath it. It expires by itself at the week reset. I am naming it
rather than pulling it: it is your ceiling and your account, and my brief
forbids me widening what this project may take.

**Without it, on this audit's arithmetic, the builder does not run again until
2026-09-14 05:23 — 5.9 days dark, the largest such loss in this project's
history — and W37's 30 free GPU-hours expire on 09-13, unspent, as the fourth
allocation in six weeks to die that way.**

**2. `D26`'s `decide_by` is today.** Both the Review (`6e79f7d`) and this audit
recommend option **(i) ATTRIBUTE THE LINE** — `pace_gate` compares against this
project's own cumulative spend from `usage_ledger.jsonl` rather than the shared
total, with the 90% stop unchanged. Priced honestly and against our own
recommendation: (i) resumes the builder on **Opus**, because `week:Fable` is at
100%, billed to the meter (i) just stopped gating. It is a Fable price for an
Opus outcome. The armed default is **(iv) MEASURE ONLY**, which is the only
*legal* default — a default may not widen a gate — and which fixes nothing.

**3. A structural question this audit raises and does not answer, because the
answer is a gate change and gates are yours.** `pace_gate` is applied to the
builder with no exemption and to the overseer with a daily one, and **not at all
to the Review or the field watch**. Of the 2.29 points/day the line recovers
against the external draw, the two unpaced/exempt document organs spend 2.0.
Whatever is decided about attribution, the ordering question is separate and
survives it: *when this project is rationed, which organ eats last?* Today it is
the only one that can move the creature. I have not proposed a number — that
would be widening or narrowing a gate, which is not mine to do.

**NO-DECISION:** items in *The audit, section by section* are a status report;
sections 1, 2, 6 and 7 are clean and there is nothing there to rule on.
