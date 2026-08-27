# OVERSIGHT — 38th audit, 2026-08-27 13:00 UTC

## VERDICT: DRIFTING — `pace_gate` has exactly ONE call site, and it is the only organ that produces ledger rows. The three organs that spend Opus on the shared meter it reads are ungated. `lib_usage.sh:51–54` diagnoses this asymmetry in its own words — *"being the only consumer with a gate, it is the one that starves"* — and the remedy shipped was a **second gate on that same only-gated consumer**. In the 48 h since the builder's last iteration the ungated organs ran **10 Opus sessions** and the gated organ ran **0 of 48 slots**.

**State.** `HEAD` is `735e63b` — the 37th audit's own commit. **Zero builder
commits since 2026-08-25 10:14.** Working tree still carries the single
untracked `experiments/tests/sm_03_nose_reports_occluded.py` (32,086 bytes,
mtime 2026-08-25 12:20). **84 PASS / 187 registered (44.9%)** — PASS unchanged
for eight days. Last builder iteration ended **2026-08-25 12:23:33**; every one
of the **48 hourly slots** since logged `PACING: … skipping`. Meters at
12:07 UTC: `week:all models` **65%** (the gate) at **47%** of the week, line
**56%**; `week:Fable` **100%** (not the gate).

**Clean results, each re-run by me rather than relayed from the 37th:**

- **§1 ledger integrity — clean.** `run verify` re-judged the record: **0
  controls run but not declared**, 0 gates that ignore their control, 0 controls
  declared but never run, 0 gates unreplayable, 0 entries unauditable. Two
  PASSes carry no control (`T0.01`, `T0.10`) — long-declared existence claims,
  a standing §1.2 note, not new. `T0.18` self-excludes correctly.
- **§2 thresholds and controls — clean.** `git log be0ecc2..HEAD` contains one
  commit and it is an audit document. I re-ran the seven-day scan across
  `registry.py`, `registry_expansion.py` and `experiments/tests/` anyway. Every
  hit is an ADDITION (the LG family, SH.02, SM.03, DP.05, NE.01) declaring new
  `control=` / `null_baseline=` / `falsified_by=` fields. The one hit that reads
  like a loosening is the opposite: `UB.10`'s `falsified_by` went *"that arm is
  EXCLUDED from"* → *"INELIGIBLE FOR"*, which is the owner's 2026-08-24
  scored-and-ineligible ruling being honoured — the arm now **runs and is
  scored** where before it could not enter. No threshold moved in the loosening
  direction, no control deleted or weakened, no `_check` gained an `or`, no seed
  count reduced, no assertion removed.
- **§5 compute accounting — internally honest.** `overruns: []`. W34 charged
  **0.3111 h of 30**, one job (`T2.15`, FAIL, harvested at `f5d8f1c`), one real
  ledger row. Every charged job reconciles against the `weeks` counter. There is
  no GPU-hour spent without a ledger entry to show for it. The books are clean;
  there is almost nothing in them.
- **§7 bakeoff hygiene — no findings.** No bakeoff has run since the 29th audit.
  `ledger.json` has not been written since 2026-08-25 10:14. No decision made
  without a learning gate, no VOID treated as a verdict, no winner inside noise.
- **The three constitutional gates all exit 0**, all run by me: **coverage** — 0
  commitments with no declared spec, 0 CLAIM-DEAD, 4 known-dangling GOAL.md
  citations on the shrink-only baseline; **decisions** — ratchet ok, 1/10
  undeclared; **champions** — ratchet ok, 6/8 seats with a phantom arena.

---

## RANK 1 — the gate's incidence is 100% on the productive organ, and its own docstring says that is the disease

The 34th blamed the wrong meter, the 35th blamed the model window, the 36th and
37th blamed the absence of a bakeoff. All four asked *whether* the gate should
exist. **None asked where it is applied.** That is a fact, not a judgement, and
it takes one grep.

### The measurement

```
$ grep -rn "pace_gate" scripts/ | grep -v lib_usage.sh
scripts/ladder_loop.sh:179:pace_gate say || { harvest_bookkeeping; exit 0; }
```

**One call site.** The other three Claude organs on this box call only
`usage_gate` — the 90% stop:

| organ | cron | model | `usage_gate` | `pace_gate` |
|---|---|---|---|---|
| `ladder_loop.sh` (the builder) | `7 * * * *` — hourly | **fable** | yes (:110) | **yes (:179)** |
| `overseer.sh` (this) | `37 */6 * * *` — 4×/day | **opus** (:47) | yes (:45) | **no** |
| `review.sh` | `37 6 * * *` — daily | **opus** (:31) | yes (:29) | **no** |
| `field_watch.sh` | `37 5 * * 1` — weekly | **opus** (:33) | yes (:31) | **no** |

The gate reads `week:all models`. That is the pool the three ungated Opus organs
draw from. The organ it throttles runs on Fable — metered separately, and
explicitly *"not the gate"* in the gate's own log line.

### What the file already knew

`scripts/lib_usage.sh:51–54`, verbatim:

> **THE CAUSE IS NOT OVERSPENDING.** `week:all models` is a SHARED pool: the
> owner's interactive sessions draw on the same meter that stops the loop … So
> the loop is stopped by consumption it does not control, **and being the only
> consumer with a gate, it is the one that starves.**

Nine lines later: *"THE FIX IS A LINE, NOT A LOWER CEILING"* — and the line was
installed on the starving party. The diagnosis and the remedy point in opposite
directions, in the same file, under the same comment block.

### What it cost, measured

Window: builder's last iteration end (2026-08-25 12:23:33) → now (12:37).

| organ | sessions in 48 h | model | output |
|---|---|---|---|
| builder | **0 of 48 slots** | — | 0 ledger rows, 0 commits |
| overseer | **8 audits** (`grep "audit start"`) | opus | 8 documents, all verdict `DRIFTING` |
| review | **2 daily runs** | opus | 2 documents |

Because the gated organ never ran, **~100% of this box's contribution to
`week:all models` in the last 48 hours was spent by organs the gate does not
touch** — producing ten Opus documents about the fact that the builder is not
running. The Review of 2026-08-27 put it in its own words: *"Every commit in the
window is an audit of the silence."*

I am not claiming on-box spend dominates the meter — the Review of 2026-08-27
measured that it does not, and I take that. **The finding does not depend on
it.** Whatever fraction the box contributes, the gate's incidence on that
fraction is 100% on the one organ that writes to the ledger.

### The consequence for B3, which is why this is RANK 1 and not a footnote

The 37th ordered the pace-gate bakeoff (`SY.01`) with two arms: **A = gate as
shipped**, **B = `JACK_NO_PACE=1`**. That arm set is right to exist and I
endorse running it — but **both arms leave the asymmetry intact**, so the
bakeoff as specified cannot discover the repair its own subject names. A race
between *"starve the builder"* and *"don't"* has no way to return *"stop
exempting the auditors."*

The docstring's diagnosis names a third arm and it costs the same as arm B —
three lines that are already written:

> **C = `pace_gate` applied to every Claude organ on the shared meter** — add
> the existing `pace_gate say || exit 0` to `overseer.sh`, `review.sh` and
> `field_watch.sh` beside the `usage_gate` line each already has.

Under arm C the builder is first in the queue for the pool rather than the only
one excluded from it, and the same three pre-registered metrics (slots run,
ledger rows, GPU-hours consumed before expiry) separate it from A and B without
any new instrument.

**I have not shown that C wins, and I am not asking for it to be adopted.** Rule
4 binds and the counterfactual is unmeasured — that is exactly what the bakeoff
is for. What I am reporting is that the arm set as ordered is incomplete.

---

## RANK 2 — the untracked SM.03 is now a LOADED failure, not just an orphan

Carried from the 36th (the hole) and the 37th (the file). What is new is that
the two have converged on the same three-day window.

- `experiments/gpu.py:274` — `git status --porcelain --untracked-files=no`. A
  brand-new test file is invisible to the GPU push guard. `protocol.py:368` asks
  the same question **with** untracked files, so the dirty stamp and the GPU
  guard disagree about what "clean" means.
- `experiments/tests/sm_03_nose_reports_occluded.py` has been untracked for
  **49 hours**. `SM.03` is registered (`f0cb81d`), is `GPU_SHORT`, and is the
  spec the builder itself named as W34's dispatch candidate.

**The loaded path:** if the gate admits an iteration and it runs
`scripts/dispatch.sh SM.03` before committing, `assert_ref_is_current` sees a
clean tree (`--untracked-files=no`), passes, and `submit()` ships a job whose
`repo_preamble(ref)` clones a HEAD that **does not contain the test file**. The
kernel dies on import. That spends the last window of W34's remaining 29.69 h on
a job that cannot run — and it is precisely the near-miss `assert_ref_is_current`'s
own docstring was written about ("a fix to UnifiedBrain existed only in the
working tree, the clone ran the published file").

*Stated as inference, not measurement:* I traced this through `gpu.py:274 → :302`
and did not run it. Verifying it costs nothing and is the builder's job.

**B1 and B2 must ship in one commit, before any dispatch.** They are one repair:
commit the file, and close the hole that made it invisible.

---

## RANK 3 — the ladder is growing in specs and shrinking in demonstrations

§3 and §8, as a number rather than a mood.

| date | PASS | registered | ratio |
|---|---|---|---|
| 2026-08-25 | 84 | 181 | 46.4% |
| 2026-08-27 | 84 | 187 | **44.9%** |

PASS has been frozen at **84 for eight days** while the registry grew by 6. Every
commit in that window added a *claim* and none added *evidence*. Six of those
specs (`LG.00/01/02/10`, `SH.02`, `SM.03`) were correct and owed work — the LG
family cleared four phantom champion arenas and a 16-day GOAL.md dangler, and
SH.02/SM.03 cleared the CLAIM-DEAD ratchet. **None of that is drift.** Every
item traces to a GOAL.md sentence: LG.00 to *"the LLM is his mouth, never his
mind"*, SH.02 to *"too cold kills him"*, SM.03 to *"olfaction … the sense that
works when sight fails"*.

The drift is the converse question, and `coverage` answers it plainly:

- **curiosity** — 12 specs, **1 passing**. GOAL.md's north star sentence.
- **one brain / unison** — 21 specs, **1 passing**.
- **fast/slow** — 8 specs, **0 passing**.
- **14 commitments** have live claim specs and **nothing passing**.

Six of the eight audit organs are green because they measure the *ladder*. The
ladder is in good order. What has stopped is climbing it.

---

## §4 the builder — alive, gated, and not at fault

Not dead, not paused, not credit-exhausted, not aborting on load. `ladder_loop.sh`
fires on schedule every hour and exits at line 179 by design. **0 iterations, 0
`rc=0`, 0 PASS delta in 48 h**, and the cause is one `if` in a library it does
not own. Nothing about the builder needs fixing.

**Projection, with its assumption stated.** The gate releases when
`pct < 25 + ceil(0.65 × elapsed)`.

- now: pct 65, elapsed 47, line 56 — gap **9 points**
- line rises at 0.65 × (100/168) = **0.387 pts/h**
- measured usage rate (08-26 14:07 = 61% → 08-27 12:07 = 65%): **0.182 pts/h**
- closing rate **0.205 pts/h** → gap clears in **~44 h ≈ Sat 2026-08-29 ~08:00 UTC**

W34's Kaggle quota (`gpu.py:_week`, `%U`, Sunday-start) expires end of **Sat
2026-08-29** — so on current rates the builder wakes with roughly **16 hours** of
GPU window left, enough for one `GPU_SHORT`. **If exogenous burn rises to
0.387 pts/h the gap never closes and the builder does not wake at all this
week.** That is the falsifiable prediction; check it against the log.

Third consecutive week of free GPU dying unspent, and the worst of the three:

| week | loop went dark | GPU-h expired unspent |
|---|---|---|
| W32 | Fri 08-14 15:07 | 8.82 of 30 |
| W33 | Fri 08-21 12:07 | 22.11 of 30 |
| **W34** (first full week under `pace_gate`) | **Tue 08-25 13:07** | **29.69 of 30, on track** |

The gate shipped 2026-08-24 to fix exactly this column. On its first full week
the loop went dark **three days earlier** than either ungated week. That is one
observation, confounded and not a verdict — it is the first row of B3's table.

---

## §6 stuck decisions — nothing overdue, nothing miscounted

All 10 armed defaults are due **2026-08-31**; none is `OVERDUE`. Nothing is
`MEANS-ESCALATED`. D13 (*"the overseer runs 4×/day on the same meter that gates
the builder"*) and D14 (*"the builder's own model is exhausted while the gate
meters a different pool"*) are the right two questions, correctly armed. I have
appended RANK 1's incidence measurement to D13 as an evidence update, because
D13's menu asks about the overseer's **cadence** and the measurement says the
sharper lever is the gate's **incidence**.

**On my standing duty to arm one entry per audit: I am declining, and the reason
is on the record.** The single `UNDECLARED` entry — *"Was physics-first retired
by argument?"* — carries **`DECIDED 2026-08-09: (a) RUN IT`** in its own body.
The 36th audit reasoned this out at `DECISIONS_NEEDED.md:2798` and concluded it
is un-armable: a default that restates the owner's answer is noise, one that
departs from it is unconstitutional. I concur and will not manufacture an arming
to satisfy a counter. The ratchet is at a floor it can only leave by **closing**,
and closing is what `decisions.py` cannot currently detect — builder item B5.

## Minor, for completeness

`gpu_budget.json` → `opening_balances["2026-W32:kaggle"].labelled_at` reads
`"2026-08-14T07:1x builder, per overseer B2"` — a truncated timestamp with a
comment fused onto it. Cosmetic, in the accounting record, harmless to every
computation (the field is never parsed). Worth one line in a commit that is
already touching the file; not worth its own.

---

## FOR THE BUILDER

**B1 + B2 go in the FIRST admitted iteration, together, in one commit, BEFORE any
GPU dispatch.** See RANK 2 — they are one repair and there is a loaded gun behind
them.

- **B1 — commit `experiments/tests/sm_03_nose_reports_occluded.py`** (eighth
  carry). Only untracked implementation of a registered spec in the repo. Do not
  freeze its gates in the same commit; the pilot has not run.

- **B2 — close the untracked hole in `assert_ref_is_current`** (36th audit RANK
  1, third carry). `experiments/gpu.py:274` drops `--untracked-files=no` so the
  GPU guard is at least as strict as `protocol.py:368`'s dirty stamp.

- **B3 — register the pace-gate bakeoff, WITH A THIRD ARM.** The 37th's `SY.01`
  design stands unchanged — tier 0, `CPU_FAST`, metrics (i) builder slots run per
  week, (ii) ledger rows per week, (iii) free GPU-hours consumed before the
  Sunday expiry; null = the measured W32/W33 rows in `lib_usage.sh`'s own table;
  `falsified_by` = the gate loses on (iii). **Add arm C** per RANK 1:
  - **A** = `pace_gate` as shipped (builder only)
  - **B** = `JACK_NO_PACE=1`
  - **C** = `pace_gate say || exit 0` added to `overseer.sh:45`, `review.sh:29`
    and `field_watch.sh:31`, beside the `usage_gate` line each already has
  - Record W34 as A's first observation: dark from Tue 08-25 13:07, **48/48
    slots skipped at time of audit, 29.69/30 GPU-h unspent**.
  - Commit the spec **before** running it. **Do not weaken or delete `pace_gate`
    on my say-so** — rule 4 binds and I have not shown the counterfactual.

- **B4 — reconcile the two spend measurements** (carried). The Review's
  transcript figure and the meter's own hourly deltas disagree, and the owner's
  levers rest on the difference. Print both, name the method for each, and say
  which the pace projection uses.

- **B5 — teach `decisions.py` to see an owner ruling written in an entry BODY**
  (carried). Report it as a distinct state — `ANSWERED-UNCLOSED` — not
  `UNDECLARED`: *"nobody asked properly"* and *"the owner answered and we did not
  act"* have opposite remedies. **Do not widen the `_SETTLED` regex.**

- **B6 — extend the pace-skip rescue path to untracked spec implementations**
  (carried, fifth time). `harvest_bookkeeping` already commits ledger rows during
  a pace skip; a registered spec whose implementation is untracked is the same
  class of orphan and would have cleared B1 automatically 49 hours ago.

- **B7 — make an `rc=0` that certifies a corpse impossible** (carried, fourth).

- **B8 — fix the 06:37 cron collision** (carried, fourth). `37 */6` and `37 6`
  both fire at 06:37; `overseer.sh` and `review.sh` run concurrently on the
  shared meter, four times a week.

- **B9 — log the model substitution as an event** (carried). `week:Fable` is at
  **100%**; the loop launches `JACK_LOOP_MODEL=fable` and will fall back to
  `opus` (`ladder_loop.sh:236`). Whatever the next admitted iteration runs on
  will not be Fable, and only the log body would ever show it.

## FOR THE OWNER

**Nothing here needs your ruling before 2026-08-31**, and I am not adding a
decision. All 10 armed defaults fire that day if you say nothing, which is the
design working.

One thing is worth your eye because it is a resourcing fact only you can see
both sides of. This box runs four Claude organs on one shared weekly meter. The
gate that decides *when* that meter may be spent is installed on exactly one of
them — the hourly builder, the only one that produces evidence — while the three
that produce documents run ungated on Opus. Over the last two days that
arrangement produced **ten Opus audit sessions and zero ledger rows**, and
**29.69 of 30 free Kaggle GPU-hours are on track to expire on Saturday** for the
third week running.

The system is now testing this rather than arguing about it (builder item B3, a
three-arm bakeoff on measured throughput). But if you already know how you want
your weekly quota split between building and watching, one line from you is
worth more than the experiment — and it would be recorded beside the
counterargument, as directives here always are.

**Neither the ladder nor the ledger is compromised.** §1, §2, §5 and §7 are
clean, and I re-ran each rather than relaying it. The honest summary of §8: we
are not closer to a curious humanoid than we were yesterday, and we are not
closer to a longer list of green ticks either — the list has not moved in eight
days. What grew is the count of documents explaining why.
