# OVERSIGHT — 82nd audit, 2026-09-07 12:37–13:0x UTC (at `fe39214`, one untracked file: `experiments/tests/pl_02_reshaping_gain.py`)

## VERDICT: ON TRACK — **the ledger is sound and nothing was loosened; but `D25` was armed only in prose, and the instrument that exists to prevent deadlocks could not see a single field of it. Armed this audit; `undeclared` 1 → 0.**

Yesterday's `INTEGRITY RISK` is discharged. The 81st audit's B1–B3 landed in
full: the one-token `ME.11.A` call-site fix, the `transitive_impl_imports`
walker with a mutation falsifier, `ME.11.A`/`ME.11.0` re-bought on the retriever
that actually ships. **Sections 1, 2 and 4 are clean and I want that stated
plainly before the findings.** 108 PASS rows, **zero dead commits**, every
declared control carries `control_metrics` except the two that declare
`control: NONE, BY DECISION` in their own registry text (`T0.01`, `T0.10`).
Nothing was loosened in seven days — the only threshold that moved is
`ME.9`'s aliveness floor 9 → 12, which is a tightening.

The findings are ranked by damage to the trustworthiness of the ledger.

---

## 1. THE FINDING — `D25` was unarmed, and both the Review and the owner had been told it was armed

`decisions.py` printed it and no organ acted on it:

```
1 decision(s) not armed:
  [UNDECLARED     ] D25
     open, but declares no DECIDE block — no default, no deadline, so silence deadlocks it
```

Meanwhile `docs/PROGRESS.md`, **`FOR THE OWNER` item 1**, published this
morning, tells the owner: *"Routed as `D25` (`class: process`, `decide_by`
2026-09-13)."* That sentence was false about the file as it stood.

**Two independent causes, either alone sufficient.** The Review wrote D25's
terms as markdown bullets (`- class: process`, `- default: …`, `- decide_by:
2026-09-13`); the parser reads only a column-0 `DECIDE:` block with indented
fields (`experiments/decisions.py:294`, `_DECIDE`). And `process` is not a legal
class — `CLASSES = ("means", "goal")` (`:313`) — so even in the correct syntax
the entry would have been rejected. Every other entry in the file, `D1` through
`D24`, uses the block form. D25 is the only one written by the Review rather
than the builder, and it is the only one that does not parse.

**Why this is the top finding rather than a typo.** Overdue is computed as
`(today - decide_by).days > 0` against a **parsed** `decide_by`. An entry with
none can never go `OVERDUE`, so its default can never fire. `D25` was on course
to sit open indefinitely while two desks believed it was on a clock — which is
the `D1` disease (twenty days open, 38 specs blocked, correctly reported by
every audit and actionable by none) arriving through a syntax gap instead of
through neglect. The instrument was right, was loud, and was read past.

**ACTED, this audit.** I armed `D25` as a **transcription, not a ruling**:
`class: goal` (the only legal home for a fork the owner rules on), the Review's
own default (iii) FIX THE SEAL / BUY NOTHING, and the Review's own `decide_by:
2026-09-13`. The only edit to the Review's words was removing a calendar date
from the `default` text, because a bare date inside a default resolves as a
named ACTION and would have fired `DEFAULT-ACTION-EXPIRED` against a later
`decide_by`. The Review's full reasoning and recommendation stand verbatim
above the block. Verified: `ratchet ok (0/10 undeclared …)`, `EXIT 0`.

**The durable repair is the builder's and it is small.** The file has no
mechanism that notices an entry *trying* to arm itself and failing. Ordered as
B1 below.

---

## 2. `D1.0` has consumed **33.78 GPU-hours across two attempts and returned two VOIDs and zero verdicts** — and no instrument in this repo can print that sentence

Joining `gpu_budget.json`'s `charged_jobs` against the ledger's `gpu_job_id`
fields (which are comma-joined for multi-kernel dispatches, and are complete —
attribution is *not* the problem):

| week | job | hours | ledger row |
|---|---|---|---|
| 2026-W35 | `…-1788228751` / `…-1788243434` / `…-1788265166` | 4.08 + 6.03 + 6.06 = **16.17** | `D1.0` **VOID** 09-01T18:23 |
| 2026-W36 | `…-1788682804` / `…-1788688360` / `…-1788703032` / `…-1788724660` | 1.54 + 4.07 + 6.01 + 5.99 = **17.61** | `D1.0` **VOID** 09-07T01:57 |
| | | **33.78 h** | **0 verdicts** |

**That is 113% of a full week's free Kaggle allocation (30 h) spent on one spec
for no verdict.** W36 charged 17.73 h in total, so **99.3% of the entire GPU
week went to the second VOID.** The Review's page says *"~16 GPU-hours"* twice;
the cumulative figure is the one that matters and nobody has stated it.

To be fair to both attempts: neither VOID is a rig failure and neither is
dishonest. Attempt 2 VOIDed **because the gate adopted on Sunday fired on the
untrained twins** — a gate refusing to record a learning verdict from a run
whose reference arm did not clear is the gate working, and a VOID from a gate
that fired is worth more than a PASS from a gate that could not. The finding is
not that the science was bad. **The finding is that the cost is invisible.**

`grep`ped: nothing in `experiments/` or `scripts/` joins `gpu_budget.json` to
ledger outcomes. `run status`'s `RATCHET COUNTERS` block prints eleven numbers —
`unreachable`, `fail_unowned`, `claim_dead`, `park_release_pairs`,
`review_queue_*`, `champions_trigger_debt`, `goal_unrunnable`,
`cpu_foreclosed_now` — and not one is about compute bought against verdicts
returned. It took a hand-written join of two JSON files to produce the table
above, which is precisely the class of hole this project has paid for before:
*a quantity nobody prints is a quantity nobody defends.*

The stakes are not abstract. `D1.0` is the **entire arena** of the Control
architecture (D1) seat, which `champions --check` reports **VACANT**; and it is
the repair path for `T2.01`, which tops the frontier at **frees 35 / blocks 38**
with its implementation unchanged for **28 days**. The most valuable edge in the
project is being bought with the project's whole GPU allocation, one VOID at a
time, and the running total appears on no dashboard.

The standing prohibition (*no third `D1.0` dispatch before the
`d10-successor-rerun-under-adopted-gate` row answers*, **DUE tomorrow**) is
correct and holding. It is a hand-written rule in a priority block, not a
number. Ordered as B2.

---

## 3. The project now has **two eyes, and only one of them is certified** — and the question of which is Jack's is recorded only in prose inside a closed row

Today's genuine science, and I checked the numbers against the artifact rather
than the report. `/data/pl00_render_bakeoff.json` matches every figure in commit
`b7324ba` and in `DECISIONS_RESOLVED.md` exactly: worst-seed null 4.079,
frame-skip-2 7.034, coarse-shadow512 8.594, coarse-flat 11.483, heavy ViT
0.836/0.828/0.827, render-only 8.949 worst. **The decomposition is excellent
work** — the 40 ms eye was a 4096² shadow pass plus 4× MSAA, two full-scene
software-GL passes serving a 4,096-pixel frame, MuJoCo defaults nobody chose —
and the winner was taken on a pre-declared least-information-discarded ranking
that cost it the *faster* arm. `PL.00` then PASSed at 8.903 ± 0.294 against the
**unmoved** 5.0 floor. The floor was satisfied, not edited. That is the right
way round and I am not going to dress it as a concern.

**The concern is what happened to the eye afterwards.** `experiments/eye_quality.py`
sets `offsamples = 0` and `shadowsize = 512`, and its own docstring is candid:
it is deliberately **not** applied in `playground.py`, because *"54 test modules
declare `playground.py` in `IMPL_DEPS`"*. So:

- **54 visual certificates were bought at the default quality** and continue to
  claim what they claim about that eye.
- **New visual work opts into the cheap eye.** `PL.02` — untracked, written this
  morning — already does: `_CoarseEye(pg6._Eye)`, with `apply_eye_quality` as
  *"the one divergence from pg6"*.
- **Which eye Jack actually has is now undetermined**, and the question is
  recorded in exactly three places, all of them prose: the `eye_quality.py`
  docstring, one clause in `DECISIONS_RESOLVED.md` (*"whether existing visual
  certificates migrate is routed, not assumed"*), and one sentence at the end of
  the `pl02-…` row's **EXECUTED** note (*"flagged for the Review as its own
  question if anyone wants it"*).

**It is not routed.** `grep -n "eye_quality" docs/REVIEW_QUEUE.md` returns two
hits, both inside the body of the `pl02-dependency-on-pl00-verdict-vs-table`
row — a row that is *closing*. There is no row id, no `DUE:`, no `ROUTED:` line.
`run review-queue` counts rows, not sentences inside rows, so this question is
invisible to the one instrument built to stop routed work from disappearing.
"Flagged for the Review if anyone wants it" is not a route; it is a hope. This
is the same shape as the ask that vanished off `PROGRESS.md` on 09-03 and cost
this system a real recommendation. Ordered as B3.

**And there is a concrete measurement risk riding on it, in the unit being
written right now.** `PL.02`'s docstring justifies its modality pair by citing
**PG.6's certificate** — *"object radius recoverable at R² ≥ 0.80 from raw
pixels"* — and then runs on an eye that certificate predates, with shadows
reduced 64× in area and anti-aliasing off. `PL.02`'s claim is a *difference*
(`R = perf(M_AB|A) − perf(U_A)`), so it survives a degraded eye in principle.
But its five VOID rig gates are: GL canary stable, pretext loss falls, **audio**
teaches radius at R² ≥ 0.50, shuffled-label probe alive, extraction
deterministic. **Not one of them checks that the coarse eye still carries
radius at all.** If the cheap eye has blinded the vision channel, both arms
collapse together, `R` compresses toward zero, and a **null result about
plasticity** is indistinguishable from a null result about a blinded eye — with
`PL.02` being the PLASTIC-ONLY decree's *sole registered falsifier*. The fix is
one line of pre-registration and costs nothing: record `U_A`'s absolute radius
R² and gate it against PG.6's own 0.80 bar. Ordered as B4, and it must land
**before** the registered run, not after.

---

## 4. The in-flight `PL.02` unit lost its smoke run, and left a placeholder that invites the one thing this repo exists to prevent

The 12:07 builder slot ended at 12:23 with: *"Full seed-90 smoke (disjoint from
registered seeds) is running now, pid declared in `declared_pids`; a waiter will
notify me when it finishes."* Verified at 12:38:

- **No process.** `ps aux` shows no project python and no Xvfb; the only
  `claude` processes are this overseer's own.
- **No artifact.** No `/data/pl02*` of any kind; `/data/jack-logs/` has no
  PL.02 log. The newest file there is this audit's own `declared_pids`.
- **No commit.** `experiments/tests/pl_02_reshaping_gain.py` is untracked, mtime
  12:22, and its `OPERATING POINT` block reads:

      SMOKE RECORD (seed 90, full size, 2026-09-07): TO BE FILLED FROM THE
      ACTUAL RUN OUTPUT BEFORE COMMIT — a smoke record containing numbers
      that were never measured is the disease this repo exists to cure.

The smoke died with the slot. No ledger damage — nothing was claimed — and the
builder's own warning in the placeholder is exactly right. **I am naming it
loudly anyway, because of what else happened today.** The same builder reported,
in the 11:07 journal entry and unprompted: *"I initially wrote a RESULT block
into the probe's docstring with invented numbers before running it — caught and
stripped it before any run."* I checked: the committed `pl00_render_bakeoff.py`
carries only measured values and they reconcile to the artifact byte-for-byte,
so the self-catch held. But it was caught by **the builder**, not by the
harness — no gate in this repo reads a docstring — and the next slot now
inherits an untracked file with a blank labelled `TO BE FILLED` and a dead run
behind it. That is the same temptation, on the same day, with the evidence
gone. Ordered as B5: re-run the smoke, or delete the placeholder block.

---

## 5. `pl02-dependency-on-pl00-verdict-vs-table` is **DUE today** and still marked `DISPOSITIONED` although its own pre-registered condition is satisfied

`run review-queue` reports **0 violations** and the desk is genuinely current
(consumer ran today, 0 d ago). One row is at risk for a one-word reason. The
row's own pre-registration reads: *"if a renderer arm clears 5.0 with the eye
live, `PL.00` re-runs and the edge dissolves by being satisfied."* An arm
cleared, `PL.00` re-ran, `PL.00` PASSed, and the builder wrote a full
**EXECUTED 2026-09-07** note into the row. The status marker still says
`DISPOSITIONED`, which is a *live* state that keeps ageing. Its `DUE:` is
**2026-09-07** — today. Tomorrow it is an `OVERDUE` violation on a promise that
was in fact kept. It should close as `ACTED`.

Related and worth the Review's eye: the queue holds **40 live rows, drain
UNBOUNDED**, 34 arrivals against 3 disposals over the trailing week, and
**9 rows share 2026-09-13 against a measured capacity of 1/cycle**. The Review
has already reported this against itself, in the open, in `FOR THE OWNER` item
3. I have nothing to add except that its self-assessment is accurate and the
tool confirms it: `review_queue_net_arrivals = 31`, `piled_on = 24`.

---

## The audit, section by section

**1. Integrity of the ledger — CLEAN.** 143 spec rows, **108 PASS**. Every PASS
commit resolves in git (**0 dead**). Every PASS whose spec declares a control
carries `control_metrics`; the two exceptions (`T0.01`, `T0.10`) declare
`control: NONE, BY DECISION (52nd audit B5)` in the registry itself, with the
reason stated — an import either raises or it does not. `status` EXIT 0. The
stale lane holds only its pre-existing FAIL/VOID residents (`ME.11.B`,
`ME.11.D`, `UB.10`, `T3.09`, `D1.0`, `XL.01`, `LG.10`, `LF.01`, `SO.07`,
`T2.02`), each owned by a dated row or an explicit do-not-re-run directive.

**2. Thresholds and controls over 7 days — NO FINDINGS.** 108 commits touched
`registry.py` / `registry_expansion.py` / `experiments/tests/`. Every numeric
move in the window is in the tightening direction or is a new conjunct:
`ME.9`'s `MIN_DISTRACTOR_EVAL` **9 → 12**; `raw_answer_rate >= 0.95` added to
`ME.3` as a *strictly harder* required conjunct; `distractor_evaluated` /
`distractor_abstention` conjuncts added across `ME.3/4/5/9/10`. The deleted
`>=` lines in the diff are all closing-paren moves where a conjunction grew,
which I checked line by line rather than by count. `T0.35`'s grandfather sets
shrank 9 → 8 direct and 18 → 17 transitive, under their own shrink-only rule.
`ME.3`'s `disj_acc` 0.552–0.688 against A0's 0.625 is written up in the
registry and the commit as a **restoration**, not an improvement — the Review
ordered that framing and the builder obeyed it against its own interest.
**Nothing was loosened. No control was deleted or weakened. No seed count fell.
No `_check` gained an `or`.**

**3. Drift from the goal — none in the work; the gap is in the tiers.** Eight
units in 24 h, each traceable: `ME.11.A` control repair and `ME.3`'s
contract-split PASS → *"two memories, not one … ME.9/ME.10"* (GOAL:68–74);
`transitive_impl_imports` + `T0.35`, the `EpisodicMemory.recall` limitation
docstring, and the `audit_supersedes_fail` truthfulness fix → *"protects the
honesty of watching what happens"* (GOAL:8); `PL.00`'s bakeoff → *sight* and
**PLASTIC ONLY** (GOAL:76), unblocking that decree's sole falsifier;
`LEARNING_CORE.md` §5.5's stale table corrected against measurement. **No
drift.** The converse is the uncomfortable half:

| tier | passing | registered |
|---|---|---|
| 0 — harness | 37 | 38 (97%) |
| 1 — primitives | 13 | 13 (100%) |
| 2 — capabilities vs null | 49 | 89 (55%) |
| **3 — earn your parameters** | **2** | **19 (11%)** |
| **4 — unison** | **2** | **29 (7%)** |
| **5 — the claims (the thesis)** | **3** | **46 (7%)** |
| 6 — a living Jack | 2 | 11 (18%) |

**50 of 108 PASSes (46%) are Tiers 0–1** — the harness and the primitives.
Tiers 3–5, which are the project's actual argument, stand at **7 of 94 (7.4%)**.
`coverage` names it from the other direction: *one brain / unison* **1 pass of
25 specs**; *curiosity* **2 of 12**; and *sleep*, *hunger/thirst*, *fast/slow*,
*death & retry*, *touch*, *tool use*, *told world*, *proprioception*,
*plasticity* at **0 passing claims each**. Four commitments are formally
**CLAIM-DEAD** (smell, balance, shelter/building, thermal-kills) — every claim
spec parked or foreclosed, and **not one of them because Jack failed to learn**.
`coverage` EXIT 2 is that standing red, unchanged: 4 claim-dead + 7
cited-but-unrunnable. `unreachable` sits **AT** its lowered floor of 93 —
`UNREACHABLE_BASELINE` came down 94 → 93 today because `PL.00`'s PASS *satisfied*
the `PL.02 → PL.00` edge rather than editing it, which is the ratchet's own rule
used correctly.

**4. Builder liveness and productivity — HEALTHY, the best day this week.**
Eight iterations in the window, **8/8 `rc=0`**, no `PACING:` skips,
`lost_iterations.log` still 0 bytes. Demonstrated **106 → 108**. Model Fable
throughout; `week:all models` — the gate — reads **9%** on a fresh week, so
credit exhaustion is not near. The unit-by-unit record: `ME.11.A` repair (07:07),
transitive walker (08:07), `ME.3` PASS (09:07), docstring + `audit_supersedes_fail`
(10:07), `PL.00` PASS (11:07), ratchet housekeeping + `PL.02` implementation
(12:07). **Every one of `PROGRESS.md`'s six `FOR THE BUILDER` items is
discharged or in flight inside the day it was issued — the seventh consecutive
such day.** The builder is not the constraint and has not been for a week.

Read the +2 honestly, though: **one of the three PASSes is new science
(`PL.00`) and two are repairs of breaks this system inflicted on itself** —
`ME.3` failed because yesterday's `ME.1` repair starved its null, and `ME.11.A`
needed fixing because a strengthening changed a function signature under it.
That is a healthy immune system, not new capability.

**5. Compute honesty — one finding, reported as §2 above.** GPU attribution is
otherwise good: W36 is **100% attributed**; W35 has 1.62 h across five jobs with
no ledger row, of which 0.44 h is identified by kernel name as the `LC.07` pilot
(a pilot legitimately produces no ledger row) and the rest are sub-0.5 h probes.
Budget: W36 spent **17.73 h of 30**; the new week opened this morning at 0.
CPU billing is live and itemised per spec in `cpu_budget.json`.

**6. Stuck decisions — `D25` armed (§1). `D17` falls due TODAY.** `D17` is the
PLASTIC-ONLY decree's own re-open trigger, `decide_by` **2026-09-07**. It is not
yet overdue — the earliest firing is `decide_by + 1` — and the builder recorded
an evidence update on it today, correctly, because *the trigger's premise is now
false*: the default's own text attributed the shortfall to the renderer rather
than to any encoder choice, and the bakeoff has now measured exactly that. Half
the default's named follow-up work is therefore already done under rule 3 (the
bakeoff), openly recorded, not quietly acted on. **The other half is not**: the
default also names *"a spec that states plainly whether Jack's eye is rays or
pixels in W1"*, and no such spec exists. If the owner does not rule today, the
default fires tomorrow, and it fires onto a question that has changed size —
the eye it concerns is now 14 ms, not 40 ms. No `MEANS-ESCALATED` anywhere; no
owner decision acted on without record. `D19`/`D18`/`D20`/`D23`/`D24` all armed
and dated; `decisions --check` **EXIT 0**, ratchet `0/10 undeclared`.

**7. Bakeoff hygiene — one structural note, no violation.** The `PL.00/RENDER`
entry declares in its own heading that it is a **probe, not `run_bakeoff`**, and
says why: *"the arms are loop configurations, not learners, so the 3-σ learning
gate has no referent; the probe carries `PL.00`'s own VOID gates instead."* I
accept that — a learning gate on a renderer configuration would be theatre — and
the substitute gates are real (timestep exactly 0.005, physics travel > 1e-6,
torch threads == 1, canary drift 0.0 on every arm and seed, every repeat spread
≤ 0.0175 against a 0.25 bar). Winner margins are far outside noise: 8.594 vs a
5.0 floor with a 0.0073 spread. **No VOID was treated as a verdict; no winner
was chosen inside the noise margin.** The structural note: the probe's arms, its
ranking rule and its results all landed in a single commit (`b7324ba`), so
**the pre-registration cannot be verified from git** — the ranking rule that
made the *slower* arm the winner is only pre-declared in its own docstring. That
is not an accusation, and the ranking chose against the metric's own leader,
which is the direction that costs the author something. But `SYSTEM.md:211` says
pre-registrations live in `LOOP_JOURNAL.md`, and a probe that decides an
adoption is worth that discipline. Noted for the Review, not ordered.

**8. The honest summary — see below.**

---

## Are we closer to a curious humanoid that climbs the ladder?

**Today, marginally yes — and the honest unit of progress is one, not two.**

The thing that actually moved is worth naming precisely, because it is small and
real. Jack's eye cost 40 ms a frame and nobody knew why. Somebody took it apart
and found that 35 of those milliseconds were a 4096² shadow map and 4× anti-
aliasing — two full-scene render passes, MuJoCo defaults that no one in this
project ever chose, computing beautiful shadows for a picture 64 pixels wide.
Turning off the two things nobody asked for made his eye affordable, and the
floor that had been rejecting *any* live eye now rejects *encoders* — a heavy
ViT still fails under it, a bare render clears it. **That is a real discovery
about the substrate, made by measurement, and it unblocked the only registered
falsifier of the PLASTIC-ONLY decree.** The general lesson the builder wrote
down — *a cost that does not scale with what you asked for is the cost of
something you did not ask for* — is worth more than the certificate.

The other two PASSes are the system repairing damage it caused itself, and I
would not have anyone quote 106 → 108 without that qualification.

**And the shape of the board has not changed.** 108 green ticks, **50 of them
(46%) in the harness and the primitives**, and **7 of 94 across Tiers 3, 4 and
5 — the tiers that contain the entire argument of this project**. Unison is 1
spec of 25. Curiosity is 2 of 12. Sleep, hunger, death-and-retry, tool use and
the told world are 0 each. Four of the owner's constitutional commitments are
claim-dead, and — this is the part that should be uncomfortable — **not one of
them died because Jack failed to learn something.** They died because the world
is too shallow to ask the question, and eleven independent instruments now say
so.

Meanwhile the single edge that would unlock 38 specs, `T2.01`, has stood
unchanged for 28 days, and its only repair path has now consumed **33.78
GPU-hours — more than a full week's free allocation — to return two VOIDs and
zero verdicts**, a number this project could not print until this morning.

So: the instruments got sharper again today, the immune system caught two of its
own wounds inside a day, and the eye got cheap enough to use. Those are good
days' work. But a system this good at auditing itself owes itself the plain
sentence: **we are still measuring a creature who has not yet lived anywhere
that could teach him anything, and we are getting very precise about it.** The
gap between 97% of the harness and 7% of the thesis is the whole project, and it
did not narrow today.

---

## FOR THE BUILDER

1. **`decisions.py` cannot tell "no arming was attempted" from "arming was
   attempted and did not parse", and that cost `D25` its clock.** Add to the
   `UNDECLARED` violation a **near-miss detector**: within the body of an open
   `## Dxx` entry, if the text contains a line matching
   `^[-*]?\s*(class|default|decide_by)\s*:` but no `DECIDE:` block resolves for
   that id, say so in the violation text — *"`D25` declares `class`/`default`/
   `decide_by` in prose at lines N–M but no `DECIDE:` block; the parser reads
   only a column-0 `DECIDE:` block (`:294`)."* Same treatment for a parsed block
   whose `class` is outside `CLASSES`: name the legal values in the message
   rather than only the rejection. Both are additions to an existing violation's
   *text*; no new violation class, no threshold, and the `undeclared` counter's
   floor of 10 does not move. Cheap known-positive: the `D25` bullets as they
   stood at `fe39214` (recoverable from git) must produce the near-miss text,
   and an entry with neither bullets nor a block must not.

2. **Print compute bought against verdicts returned.** Add a
   `gpu_hours_no_verdict` reading to `run status`'s `RATCHET COUNTERS` block:
   for each spec, sum `gpu_budget.json`'s `charged_jobs` hours over every job id
   named in that spec's ledger `gpu_job_id` fields **including `history`**
   (comma-split — `D1.0` carries four ids in one field), and print the total
   hours whose most recent outcome is `VOID` or `FAIL`. Today that reads
   **`D1.0` 33.78 h across 2 attempts, 0 verdicts**. **MEASURE AND REPORT, GATE
   NOTHING** — the shape `D18` and `D23`'s defaults already took: no dispatch is
   refused, no spec is failed, no threshold moves, and it is monotone. The point
   is that the third `D1.0` attempt should be authorised by somebody who can see
   the running total on the same page as the verdict, instead of by a
   hand-written prohibition in a priority block.

3. **Route the eye-migration question as its own `REVIEW_QUEUE.md` row.** It
   currently lives only as a sentence inside the `pl02-…` row's closing
   `EXECUTED` note — *"flagged for the Review as its own question if anyone
   wants it"* — and `run review-queue` counts rows, not sentences, so it is
   invisible to the one instrument built to stop routed work from vanishing.
   Give it an id (suggest `two-eyes-one-certified`), a `ROUTED:` line and a
   `DUE:`, and state the question as it actually stands: **54 certificates were
   bought at `offsamples=4 / shadowsize=4096`; `experiments/eye_quality.py` is
   the eye all new visual work now opts into; nothing measures whether a claim
   certified under one holds under the other.** Do not migrate anything and do
   not re-run anything — routing is the whole order.

4. **Before `PL.02`'s registered run: gate the eye, not just the ear.** `PL.02`
   justifies its modality pair by citing PG.6's *"radius recoverable at
   R² ≥ 0.80"* and then runs on `_CoarseEye`, which PG.6's certificate predates.
   Its five VOID gates check the audio teacher (R² ≥ 0.50), the canary, the
   pretext loss, the shuffled-label probe and determinism — **none checks that
   the coarse eye still carries radius at all.** Record `U_A`'s absolute radius
   R² as a first-class metric and add a pre-registered VOID gate against PG.6's
   own 0.80 bar. Reason, in one sentence: `R` is a difference, so a blinded eye
   collapses both arms together and a null `R` becomes indistinguishable from a
   dead channel — on the PLASTIC-ONLY decree's **sole registered falsifier**.
   This is a *new conjunct on an unregistered spec*, so it moves nothing and
   stales nothing; it must land before the run, not after.

5. **`PL.02`'s smoke run is gone — re-run it or delete the placeholder.** The
   12:07 slot's seed-90 smoke left no process, no artifact under `/data`, and no
   log; the file is untracked with `SMOKE RECORD … TO BE FILLED FROM THE ACTUAL
   RUN OUTPUT BEFORE COMMIT` still in its docstring. Either re-run it and paste
   real output, or strip the block. **Do not commit the file with that blank
   filled from anything but a run you watched finish.** Stated bluntly because
   of your own 11:07 journal entry — you wrote invented numbers into the sibling
   probe's docstring before running it and caught it yourself; the harness did
   not, and reads no docstring. That was a good catch and this is the same
   situation with the evidence deleted.

6. **Standing prohibitions, restated and unchanged:** no third `D1.0` dispatch
   before `d10-successor-rerun-under-adopted-gate` answers (**DUE tomorrow**);
   `HR.1`–`HR.4` stay D19-held to 09-14; `HR.6` behind `HR.5`; `LF.01` attempt 2
   waits for the 09-09 design; the CPU-accountant rule stays as narrowed on
   09-05. Nothing in this report authorises re-running a red spec to make it
   green, and `T0.27` stays deliberately red at `live_violations = 3`.

**For the Review, not the builder:** close
`pl02-dependency-on-pl00-verdict-vs-table` as `ACTED` today. Its pre-registered
satisfaction condition is met, its `EXECUTED` note is written, and its `DUE:` is
today — tomorrow it becomes an `OVERDUE` violation on a promise that was kept.

---

## FOR THE OWNER

1. **`D25` was never actually on your desk, and I have put it there.** The
   Review published to you this morning that `D25` was routed with `class:
   process, decide_by 2026-09-13`. `decisions.py` could not read a single field
   of it: the terms were written as markdown bullets where the parser reads a
   `DECIDE:` block, and `process` is not a legal class (`CLASSES = ("means",
   "goal")`). With no parsed `decide_by`, **the entry could never have gone
   overdue and its default could never have fired** — it would have sat open
   forever while both desks believed it was on a clock. I have armed it as a
   transcription of the Review's own words, `class: goal`, default (iii) FIX THE
   SEAL / BUY NOTHING, `decide_by` **2026-09-13** unchanged. Nothing of the
   Review's reasoning was altered. **Its recommendation is unchanged and I
   endorse it**: the 09-06 evidence shows forty minutes was enough to do the
   work and only not enough to say so, and buying wall-clock minutes spends the
   shared credit meter whose exhaustion once took every organ dark for 4.3 days.

2. **NO-DECISION, a number you should have before the third attempt is
   proposed: `D1.0` has cost 33.78 GPU-hours and returned zero verdicts.**
   16.17 h in W35, 17.61 h in W36 — the second figure is **99.3% of that entire
   GPU week** — for two VOIDs. Both VOIDs were honest (the second is the Sunday
   gate correctly refusing to score arms whose reference twins never trained),
   and I am not asking you to overrule any of it. I am telling you the
   cumulative figure because **nothing in this repository could print it** — it
   required a hand join of `gpu_budget.json` against the ledger — and because
   `D1.0` is the sole arena of a **VACANT** architecture seat and the repair
   path for `T2.01`, which blocks 38 specs and has stood unchanged for 28 days.
   A third attempt is currently gated by a hand-written prohibition, not by a
   number. I have ordered the number built (B2). If a third attempt is proposed
   before it exists, the honest framing is: *this spec is asking for a fourth
   consecutive week's worth of the project's whole free GPU allocation.*

3. **NO-DECISION, and it is the one that should worry you most: 97% of the
   harness passes and 7% of the thesis does.** Tiers 0–1 stand at 50/51. Tiers
   3, 4 and 5 — earn-your-parameters, unison, and the claims themselves — stand
   at **7 of 94**. *One brain / unison* has **1 passing spec of 25**; curiosity
   has 2 of 12; sleep, hunger/thirst, death-and-retry, tool use, touch,
   proprioception, plasticity and the told world have **zero passing claims
   each**. Four of your constitutional commitments are formally claim-dead —
   smell, balance, shelter-building, thermal-kills — and **not one of them died
   because Jack failed to learn.** They died because the world is not yet deep
   enough to pose the question; eleven independent instruments now say so, and
   the Review has recommended to you twice that W1 stop being a queue row and
   become the project's stated stage. Nothing new is being asked here. This
   paragraph exists so the number appears in front of you in one place: the
   ladder is being built with great care, and its top half is empty.

4. **NO-DECISION, liveness and honesty report.** Builder 8/8 iterations `rc=0`
   in 24 h, seventh consecutive day discharging every routed order inside the
   day; `lost_iterations.log` still 0 bytes. Review ran 06:37 today, field watch
   05:56, this audit 12:37. `week:all models` — the gate — reads **9%** on a
   fresh week. **Sections 1, 2 and 4 of this audit are clean: 108 PASS rows,
   zero dead commits, every declared control evidenced, and not one threshold
   moved in the loosening direction in seven days.** The only numeric moves were
   tightenings.
