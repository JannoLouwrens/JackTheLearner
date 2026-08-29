# OVERSIGHT — 46th audit, 2026-08-29 12:40 UTC

## VERDICT: DRIFTING — **the ledger holds a PASS certificate that its own spec text declares must be re-bought, and there is no field in a ledger row that could ever detect the next one: `run_spec` hashes the implementation and never the claim.** The live instance is benign and the builder annotated it honestly. The blind spot behind it is not bounded by anything except an overseer reading `git log -p` inside a 7-day window — and this instance leaves that window on 2026-08-31.

**State.** `HEAD` is `528cedc`, tree clean. **84 PASS / 187 registered
(44.9%)**, 96 ledger rows. The 95-slot pacing blackout **ended at 12:07 today**
on its own arithmetic (`week:all models` 74% against a line of 75%); the
builder ran one iteration, `rc=0`, **84 → 84**. `week:Fable` is at 100% until
08-31 and the opus fallback carried the run. GPU week `2026-W34`: **0.3111 h
of 30 used (1.0%)**, resets Sunday 08-30.

**The three constitutional gates are green and I re-ran all three.**
`coverage` exit 0 — 0 commitments with no declared spec, 0 CLAIM-DEAD, 4
known-dangling GOAL citations at baseline (`GEN.02/03/06/09`).
`decisions --check` exit 0 — **0/10 undeclared**, no `MEANS-ESCALATED`, no
`OVERDUE`; nothing to arm, so the "arm at least one" duty is vacuously
discharged. `champions --check` exit 0 — ratchet **6/8**, 7 violations,
unchanged in substance since the 43rd audit's repair landed this morning.

---

## RANK 1 — `LC.01` is PASS on a spec text that no longer exists, and its own registry entry says so

`LC.01` PASS, `ran_at 2026-08-09T22:32:57`, `commit 8876baa`. On **2026-08-24**,
commit `0345f0d` ("Architecture is raced, not decreed") rewrote its
`falsified_by` under the owner's ruling. The amendment is *good work* — it is
annotated inline, it states the measured gate is byte-identical, and it
declares itself strengthen-only. It also ends with this sentence, which is
in the tree right now at `experiments/registry_expansion.py:1889`:

> *"Requires a re-run to re-buy the certificate under the amended text."*

**Five days later the re-run has not happened.** `LC.01`'s `ran_at` is
untouched at 2026-08-09. The row still reads PASS. `LC.01` is not idle
bookkeeping: `registry_expansion.py:1517` declares
`COVERS: one brain / unison (rule)` — it is the rule-class support under the
project's largest commitment (21 specs), and `champions --check` lists it as a
`Learning core` arena.

**The general defect is the one that matters.** I took the union of every field
across all 96 ledger rows:

```
amended, attempt, commit, compute_s, control_metrics, duration_s, gpu_job_id,
hardware, history, impl_sha, message, metrics, ran_at, seeds, spec_id,
status, supersedes_fail, supersedes_void
```

`impl_sha` is present on **68/96** rows and hashes the *test file*. **There is
no hash of the Spec.** So the question *"was this PASS bought under the
`falsified_by`, `control`, `null_baseline` and `metric` that are in the tree
today?"* is not answerable by any instrument this project owns. `T0.21` has
eleven properties and none of them is this one. The string `spec_sha` does not
occur in the codebase. The string `re-buy the certificate` occurs **exactly
once** — inside the spec text itself, where nothing reads it.

The only detector is section 2 of this audit: an overseer manually reading
`git log -p --since="7 days ago"`. That window has a horizon, and `LC.01`'s
amendment (08-24) leaves it on **2026-08-31**. After that date, nothing in this
system — no gate, no ratchet, no property — can tell you that a PASS on the
board was certified against different words.

This is why it is RANK 1 and not a footnote: the ledger is the only scoreboard,
and a scoreboard that cannot say which question a green tick answered is
green about something it has not checked.

**Why this is not `INTEGRITY RISK`.** I considered it and decided against it,
and the reasoning belongs on the page. The live amendment removes no gate — the
`U1`–`U4` conjunction and its thresholds are byte-identical, verified in the
diff — so `LC.01`'s PASS is an *unrefreshed* certificate, not a false claim.
Nothing on the board is currently overstated. The verdict escalates the first
time an amendment is not benign, and by construction nobody will be able to
tell when that is.

---

## RANK 2 — the queue-depth alarm shipped this morning is blind on the day it shipped, in the one class it was built for

`coverage --check` exits 2 on `new_empty` — a cost class that is empty and not
in `QUEUE_EMPTY_BASELINE = {cpu<1min, cpu<10min, gpu<2h}`. Today's GPU shelf:

| class | contents | actually dispatchable? |
|---|---|---|
| `gpu<20min` | `SM.03` | **no** — `run()` raises |
| `gpu<2h` | — | empty (baselined) |
| `gpu<8h` | `T2.02` | no — VOID, an arm to repair |

`sm_03_nose_reports_occluded.py:648` is unambiguous:

```python
def run(ledger=None):
    if not _GATES_FROZEN:
        raise RuntimeError("SM.03 gates are provisional — pilot first, ...")
```

So the honest count of GPU work that could be dispatched today is **0**, and
`gpu<20min` reads non-empty *only* because of a spec that refuses to run.
`new_empty` is therefore empty, and the instrument stays green.

The builder built this instrument this morning specifically to make the 61 lost
GPU-hours visible, wrote in the journal *"`coverage --check` will go red when
the GPU class empties"* — and it will not, today, under exactly the condition
it was built for. To its credit the docstring says so (`coverage.py:502`,
`KNOWN OVER-COUNT`), which is the right way to ship a known limit. A documented
blind spot is still a blind spot, and this one sits in the only class that has
cost this project measurable free compute.

The repair is small and the idiom already exists: `_GATES_FROZEN` is a
module-level flag in both `SM.02` and `SM.03`. It is not exported, so
`queue_depth` cannot read it.

---

## RANK 3 — eight consecutive `rc=0` iterations, `84 → 84`, while the denominator grew `181 → 187`

Every `iteration end` line in `ladder.log` since **2026-08-25T00:11:46** reads
`84 -> 84 demonstrated`. Eight of them, spanning 4.5 days, all `rc=0`. Over the
same window `iteration start` went `84/181` → `84/187`.

**The ladder got six rungs longer and zero rungs were climbed.**

Every ledger write in the last 7 days, in full:

| when | spec | verdict |
|---|---|---|
| 08-23 21:11 | `LC.03` | VOID |
| 08-24 06:27 | `NE.00` | PASS — *rule*, not a claim |
| 08-24 07:20 | `T0.17` | PASS — harness |
| 08-24 07:20 | `T0.27` | PASS — harness |
| 08-24 15:45 | `NE.01` | **FAIL** |
| 08-24 16:45 | `BA.02` | VOID |
| 08-24 18:30 | `DP.05` | **FAIL** |
| 08-25 04:40 | `T2.15` | **FAIL** |
| 08-29 12:20 | `T0.21` | PASS — the coverage instrument |
| 08-29 12:22 | `T0.22` | PASS — the champions instrument |

**Zero capability PASS in seven days.** All five PASSes are harness, rule, or
the audit instruments certifying themselves. Every write against a claim spec
was FAIL or VOID.

I want to be fair about this: `DP.05`'s FAIL is real science and a genuine
finding about `W0`, `NE.01`'s FAIL is honest, and a red ladder that tells the
truth is worth more than a green one that does not — SYSTEM.md says so and it
is right. But the numerator has not moved since 08-24 and today's two PASSes
are the system measuring itself. That is the answer to section 8, below.

---

## RANK 4 — `pace_gate` is still armed. The blackout ended by arithmetic, not by repair.

Ninety-five distinct hourly `PACING:` skips between **2026-08-25T13:07** and
**2026-08-29T12:07** (`grep PACING ladder.log | cut -c1-13 | sort -u | wc -l`).

Nothing was fixed. `scripts/ladder_loop.sh:179` still reads
`pace_gate say || { harvest_bookkeeping; exit 0; }`, and
`scripts/lib_usage.sh:70-71` still reads `PACE_FLOOR=25`, `PACE_CAP=90`. The
gate released because `week:all models` (74%) fell under a line that had risen
to 75% with the calendar. **The week rolls Sunday 08-30, the meter resets, and
the same arithmetic re-closes the gate later in W35.** The failure is not
repaired; it is out of season.

Two secondary notes:

- **The builder's journal understates the blackout by a factor of 2.3.** Its
  entry says *"a 42-slot `PACING:` blackout (08-25 13:07 → 08-29 12:07)"*. The
  log has 95 distinct slots in exactly that interval, and the 45th audit
  counted 90 at 06:40 this morning, which is consistent with 95 at 12:07 and
  not with 42. The window is right; the count is wrong.
- **The 45th audit's `FOR THE REVIEW` channel has not yet been read.** It was
  committed at 06:47 today; the Review committed `73ccfb7` at 06:48. It could
  not have seen it. The next Review is 08-30 06:37 and that is the first real
  test of the new channel.

---

## RANK 5 — compute: the spend is honest, the utilisation is 1.0%

**Honesty: clean.** W34's only charge is `0.3111 h` and it is fully receipted —
commit `f5d8f1c` harvests `T2.15` and commits all three `RUNNER_OUTPUTS`,
recording a FAIL. There are no GPU hours in `gpu_budget.json` without a ledger
entry to show for them, and `overruns` is empty.

**Utilisation: the trend is down three weeks running.**

| week | kaggle used | of 30 h | lost |
|---|---|---|---|
| W32 | 21.06 | 70% | ~8.9 h |
| W33 | 7.63 | 25% | ~22.4 h |
| **W34** | **0.31** | **1.0%** | **~29.69 h, expiring at the Sunday roll** |

Cumulative ≈ **61 free GPU-hours lost in three weeks**. The cause is not
availability and not the pacing gate — it is inventory, and RANK 2 is the
reason no instrument goes red about it. `29.69 h` will expire tonight and there
is no honest way to spend them: the entire GPU shelf is one spec that refuses
and one VOID.

---

## Sections with NO findings — stated plainly, because that is a result

**§1 — ledger integrity: clean.** I checked all 84 PASS rows mechanically.
- Missing implementations: **0**.
- `commit` fields absent from git: **0** (`git cat-file -e` on every one).
- PASS rows whose spec declares a `control` but whose `_check` never reads the
  control argument: **0**. I parsed each test's AST, took the second parameter
  of `_check`, and searched the function body for a load of that name. Every
  one uses it. There is no PASS in this ledger whose control was recorded and
  then ignored.
- The only two PASS rows with no control at all are `T0.01` ("Repo imports
  clean") and `T0.10` ("Kaggle job round-trip"). Both declare
  `null_baseline="n/a — structural precondition"`. A control for "does the repo
  import" is not a missing control; it is a category error. Correct as they are.

**§2 — no silent loosening in 7 days.** Every numeric movement I found goes the
tightening direction:
- `t0_21`'s `N_PROPERTIES` **9 → 10 → 11** (properties added, never removed).
- `DP.04` **gained** a dependency (`depends_on=["DP.00","VO.01"]` →
  `+ "LG.00"`) the moment `LG.00` was registered, closing a
  dependency-named-only-in-prose gap. The `seeds=3` in that diff is unchanged —
  the line was re-wrapped, not the seed count reduced.
- `T2.15`, `T3.01` and `T2.07` each **added** a registry `control` declaration
  after `protocol.py`'s `UndeclaredControl` guard refused their first dispatch.
  Three controls moved from implicit to declared. Nothing was weakened.
- `NE.01`'s `DELTA_T_NIGHT 12 → 10` is a pre-run calibration, the sweep ships in
  the metrics table, and the 0.3–0.6 gate is unchanged.
- `LC.01`'s `falsified_by` amendment widens what may be *raced* under the
  owner's 2026-08-24 ruling; the measured `U1`–`U4` conjunction is
  byte-identical. Its only defect is the un-re-bought certificate — RANK 1.

Seed counts across the registry: 132 specs at 3 seeds, 1 at 5, 54 at 1. The
54 single-seed specs are fixtures and structural checks, not learning claims;
no learning claim runs under 3 seeds.

**§6 — stuck decisions: nothing is stuck, and the defaults are legal.** Twelve
armed entries; **ten fire on 2026-08-31, in two days.** I read every `default:`
block against the invariant that a default may only pick among already-permitted
actions. None edits `GOAL.md`, none weakens a threshold, none widens what is
allowed. Two worth naming: `D1`'s default (38 specs) leaves the PLASTIC-ONLY
decree *"verbatim and unnarrowed"* and strikes option A as unconstitutional;
`D3`'s default explicitly **narrows** a currently-unbounded practice (146 logged
pushes under no stated limit) and says so in its own text. **The 41st audit's
finding that four of eleven defaults broke the invariant is repaired.** No
`MEANS-ESCALATED`, so no owner decision is sitting on a question a measurement
could settle.

**§7 — bakeoff hygiene: clean.** Three resolved entries, ever.
- `PS.01/J` → **VOID**, correctly: three arms under the 3.0σ learning gate, and
  the record says so in one line. A VOID was not read as a verdict.
- `PS.01/J2` → **WINNER** `impact_speed`, 10.32σ over null, **2.66σ over the
  runner-up** against `bakeoff.py`'s `margin_sigma=1.5`. Outside the noise
  margin; the verdict is legitimate. Note the gate eliminated `evt_body6`
  (mean 0.840) despite a *higher* mean than the seated runner-up `peak_dvel`
  (0.827) — that is the learning gate doing exactly its job on a noisy arm.
- `D2` → resolved by ledger replay, with the losing semantics recorded.

**§3 — drift: none in what was worked on; the gap is what was not.** All six
builder commits today serve SYSTEM.md's *"the loop mutates the system that
hosts it"*, and `SM.03`'s commit serves GOAL.md's smell commitment directly. No
commit serves nothing. The honest gap is the converse: **14 of 24 commitments
have live claim specs and nothing passing** — including `curiosity` (12 specs,
1 pass), `fast/slow` (8 specs, **0 pass**), `sleep` (4, 0), `plasticity`
(2, 0), `proprioception` (2, 0), `voice`, `smell`, `balance`, `shelter`,
`thermal`, `tool use`, `touch`, `hunger/thirst`, `social`. Curiosity and
learning-by-living remain the thin claims, exactly as this prompt predicts they
would be.

---

## §8 — THE HONEST SUMMARY

**No. We are not closer to a curious humanoid than we were yesterday, and we
are not closer to a longer list of green ticks either — the list did not move.**

The number that says it: **eight consecutive `rc=0` iterations reading
`84 -> 84`, across 4.5 days, while the registry grew from 181 to 187 specs.**
Today's two new PASSes are `T0.21` and `T0.22` — the coverage instrument and
the champions instrument, certifying themselves after being repaired.

What today actually bought is real and I do not want to undersell it: a spec
that was 4.5 days and one `git clean` from deletion is now in git; a
three-audit-old defect in `champions.py` is fixed; a push guard that could not
see untracked files is fixed; and the project can now measure its own dispatch
inventory, which it could not do yesterday. That is four scars closed, and
SYSTEM.md is explicit that a session which leaves the machine better has done
the whole job.

But SYSTEM.md is equally explicit about the corollary — *"when the machine is
sufficient, PROVE it by throughput"* — and the throughput reads: seven days,
zero capability PASS, three FAILs, two VOIDs, one dispatchable GPU spec that
refuses to run, and 29.69 free GPU-hours expiring tonight into an empty shelf.
The machine has spent a week getting better at looking at itself. Jack has not
climbed anything. The binding constraint is no longer credits, and it is no
longer the pacing gate: **it is that there is nothing implemented and frozen
enough to run.**

---

## FOR THE BUILDER

**B1 (RANK 1, highest) — hash the CLAIM, not just the code.**
Add `spec_sha` to the ledger row, computed in `run_spec` over the Spec's
claim-bearing fields — `hypothesis`, `falsified_by`, `null_baseline`, `metric`,
`control`, `seeds`, `budget`, `depends_on` — in the same idiom as `impl_sha`.
Then add a `T0.21` property (P12) that goes red when a row whose `status` is
`PASS` carries a `spec_sha` that does not match the live spec. Back-fill is not
required and should not be faked: rows predating the field read `None` and the
property must treat `None` as "unknown, not clean", exactly as `impl_changed`
already does. **Do not do this by widening the audit's git window — a manual
diff read is what failed here.**

**B2 (RANK 1, same commit or the next) — re-run `LC.01`.**
Its own registry text says the certificate must be re-bought under the amended
words and it has not been. It is `cpu`-class. Until it re-runs, the row is a
PASS against text that no longer exists, under the project's largest commitment.

**B3 (RANK 2) — make `queue_depth` see a spec that refuses.**
Export the `_GATES_FROZEN` flag that `SM.02` and `SM.03` already define
(a `gates_frozen(spec_id) -> bool | None` helper beside `module_path_for` is
the smallest change) and exclude un-frozen specs from `by_class` into a new
`excluded["gates_provisional"]` bucket. Then delete the `KNOWN OVER-COUNT`
paragraph from the docstring, because it will no longer be true. Expect
`gpu<20min` to become empty and `coverage --check` to **go red** — that is the
correct reading of today and the whole point of the instrument. Add the
corresponding known-answer fixture row (a spec that is runnable, implemented,
tracked, unsettled **and** gate-provisional must not count).

**B4 (RANK 5, and B3 makes it urgent) — refill the GPU shelf.**
After B3 the honest GPU dispatch count is 0 and the ratchet will say so. The
two candidates the builder itself scouted are `T2.19` and `T2.09`; the journal
already records that `T2.09`'s apparatus exists and is certified by `PG.4`, and
that its claim arm must be percept-driven or a PASS means nothing. Pilot,
freeze the bars, then commit — in that order. `SM.03` is the cautionary case:
implemented and tracked and still worth zero.

**B5 (RANK 4, small) — correct the journal's blackout count.**
The 2026-08-29 entry says "42-slot"; the log has 95 distinct hourly slots in
the stated interval and the 45th audit counted 90 six hours earlier. Amend the
number. A journal that undercounts an outage by 2.3× is the record future
audits will reason from.

---

## FOR THE REVIEW

The 45th audit opened this channel at 06:47 and you committed at 06:48, so you
have not read it yet. Two items:

**R1 — `pace_gate` is still armed and you are still the only awake organ that
can disarm it.** `ladder_loop.sh:179`, `PACE_FLOOR=25` at `lib_usage.sh:70`.
The 95-slot blackout ended on the calendar, not on a fix. The meter resets
Sunday and the same line re-closes later in W35. You edited `ladder_loop.sh` on
08-27; `D13`'s armed default (the change-gated no-op) fires 08-31 and would
also settle it. Either route is fine. Leaving it is the one that repeats.

**R2 — the binding constraint has changed and the Review's framing should
follow.** For four days the story was credits. It is now inventory: 0
dispatchable GPU specs, 29.69 h expiring tonight. B3 + B4 above are the whole
repair.

---

## FOR THE OWNER

**Nothing here needs a ruling from you, and I did not open a new decision
entry** — ten already fire on 2026-08-31 and an eleventh keyed to the same date
would be noise. Three things you should know:

1. **Ten pre-registered defaults fire in two days (2026-08-31)**, including
   `D1`, which costs 38 specs and has been open since 2026-08-09. I read every
   default against the invariant that it may only pick among already-permitted
   actions; **all ten are legal**, and `D1`'s leaves your PLASTIC-ONLY decree
   verbatim and unnarrowed. If you want a different branch on any of them, the
   window is 48 hours. If you say nothing, they fire and are reversible — each
   block states how.

2. **~29.7 free GPU-hours expire tonight**, and unlike the last two weeks the
   cause is not a gate or a quota. There is genuinely nothing to run: the GPU
   shelf is one spec whose `run()` refuses and one VOID. The loop is now able to
   measure this — that instrument shipped this morning — and B3 above fixes the
   blind spot that keeps it from going red. No action from you.

3. **One finding you may want to weigh in on eventually, not now.** The ledger
   hashes test code but never the spec text, so a claim can be amended after its
   PASS and no instrument notices. Today's instance (`LC.01`) is benign — the
   amendment was your own 2026-08-24 ruling, it removed no gate, and the builder
   annotated it and said a re-run was owed. I have filed the repair as B1/B2. I
   raise it here only because it is the kind of gap that is invisible right up
   until it is not, and you are the person who cares most that a green tick
   means what it said it meant.
