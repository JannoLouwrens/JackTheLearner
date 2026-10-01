# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-10-01 00:3x–00:5x UTC — the 134th audit.** Six hours after the 133rd, on
cadence. My window is the builder's slots `19:07` through `00:07`: **six slots,
four `rc=0`, one ABORT before launch, one `rc=1` lost to session limits.**
Demonstrated moved **107 → 107**.

Instrument exit codes, re-derived this sitting from source and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run status` **2**, `run review-queue` **2**.

> **Disclosure about my own method, because it is exactly the shape of thing
> this organ is supposed to catch:** my first `run status` call read **`rc=120`**.
> That was my own `| head -120` closing the pipe — Python exits 120 on
> `BrokenPipeError` at shutdown — not the tool. I re-ran it to a file and got
> `2`. A successor inheriting `rc=120` from a transcript would be inheriting an
> artefact of the reader. (`docs/LESSONS.md`: never tail an instrument; the same
> applies to heading one.)

Ratchet delta, quoted as a DELTA from `run status`'s own SLOT LINE before
anything here is recorded: **2 MOVED — `review_queue_violations` 8 → 14 and
`review_queue_violation_forms` gained `OVERDUE: 6`; no counter refused to
compute; floors 3 ABOVE (`decisions_default_action_expired`,
`pass_on_dead_dependency`, `unreachable`), 0 BELOW, 0 UNVERIFIED.** The move is
a CLOCK move — six dated rows reached their date — and the tool says so itself:
*"a real event with a real owner, and no commit is to blame for the rise."* All
three floors are inherited, each has a written cause I re-derived, and none is
mine.

Ledger: **PASS 107 / FAIL 33 / VOID 16 / BLOCKED 1** over 157 rows with a
verdict against 254 specs registered → **42.1 %**. Rework: **127 of 157 rows are
attempt > 1 (80.9 %)**.

**Tree state: CLEAN at my open and clean when I committed.** I verified
`git status --porcelain` empty at both ends, and my own commit `git add`s by
name. Nothing was held dirty while this page was drafted.

**What I did NOT do, named rather than omitted.** I re-ran no spec (the
no-dry-run rule) — every number below is read from the committed ledger, from
git, from `/data/jack-logs/ladder.log`, or computed by calling a spec's
**read-only** scanner (`audit_supersedes_fail`, `usage_attribution.classify`)
and never its `run()`. I dispatched nothing. I armed no decision:
`decisions_undeclared` reads **0** — there is nothing armable on the register,
and manufacturing an entry to satisfy the per-audit quota is the disease the
quota exists to prevent. I fired no default: `D37` is the only armed entry and
its date (2026-10-04) has not arrived. I escalated nothing new to the owner;
every finding below is a means question, routed to the builder.

---

## VERDICT: DRIFTING — and the specific thing that got worse is that **the honesty machinery itself shipped two false-assurance defaults in 36 hours.** A brand-new admissibility check published `admitted | yes` for the exact arm it was built to exclude; a one-day-old liveness counter read `0` through the slot the builder actually lost. Neither was a lie and both were disclosed somewhere — **but in both cases the correction lives in a docstring, a commit message or a queue row, and the thing a reader LOOKS at renders the uncorrected claim.** That is now this project's dominant failure mode, three instances deep, and yesterday's `kill_scope` fix is the precedent for all of them. Separately, and unchanged: **378 commits in seven days, zero lines of Jack**, and the creature gate has read `NONE` for fifteen consecutive slots.

Why not `INTEGRITY RISK`: the ledger's mechanical integrity held on every check
I ran (section 1 — 107/107 implementations resolve, 0 dangling commits, 0
undeclared controls), **no threshold moved in the loosening direction**
(section 2, a real result), and no claim today rests on either defect — `SO.10`
is `FAIL` with its seat VACANT, and the lost slot cost an hour, not a verdict.
These are findings about what the board RENDERS. That is serious and it is not
the same as a corrupted ledger, and I am not inflating it into one.

---

## RANK 1 — `bakeoff.py`'s new admissibility predicate defaults to `admissible = True`, so a spec that supplies none renders every arm as admitted. **Twenty-one minutes after the predicate landed, a mechanical regate sweep published `SO.10 — TIE — laplace-full … admitted | yes` into `docs/DECISIONS_RESOLVED.md` — `laplace-full` being the precise arm the desk ruled INELIGIBLE 36 hours earlier, in the ruling that commissioned the predicate.** (HIGH — a fresh honesty mechanism actively certifying the case it was built to catch, on the page this audit's section 7 exists to read)

### What landed, and it was good work

The `so10`+`lg13` paired ruling was executed in full at `1b651ff`/`e344e88`
(22:22–22:26), **eleven days before its 2026-10-11 DUE**. `run_bakeoff` now
takes a spec-supplied `admissible=` predicate evaluated after scoring and before
ranking; an inadmissible arm is scored-and-ineligible and **not ranked**, so it
cannot win, tie, break a tie on cost, or VOID the field; a field with fewer than
two admissible arms VOIDs at the door. A 9/9 battery replays `SO.10`'s shape in
both directions. The design is right and it answers the row's question.

### The finding

Two lines of `experiments/bakeoff.py` decide what an unsupplied predicate means:

```python
:136    admissible: bool = True
:245    admissible=(admissible is None or bool(admissible(arm)))
```

So **"no predicate supplied" renders identically to "predicate says yes."**
`experiments/tests/so_10_trust_rule_bakeoff.py` supplies no `admissible=` — I
grepped it; `run_bakeoff` is called at `:268` without one — and the ruling
explicitly and correctly says a re-run is **"NOT owed"**, so the builder was
right not to wire it. The consequence nobody priced is downstream of the
*render*, not the run:

```
## SO.10 — TIE — laplace-full
laplace-full leads laplace-w30 by only 0.26 sigma (margin 1.5). The choice
does not matter yet; taking the cheapest tied arm (laplace-full, cost 0).

| arm          | mean  | sigma over null | gate | admitted | cost |
| laplace-full | 0.722 | 5.79            | pass | yes      | 0.0  |
```

Provenance, traced: that block was appended by **`5bc7471` (22:43) — "regate
sweep: cheap stale certificates re-bought mechanically."** Nobody chose to
publish it. The sweep re-bought a stale certificate, the new renderer added an
`admitted` column, the default filled it with `yes`, and the page now carries
the defect's own verdict wearing the new machinery's endorsement. It is the
**third and newest** `SO.10` record on that page (there are three, at
`:1986`, `:2016` and above), **none** annotated.

What the desk actually ruled on 2026-09-29, which appears nowhere near those
records: the Person-model seat **stays VACANT**, `SO.10`'s **FAIL stands**, the
two *eligible* arms tie at 0.16 sigma with equal declared cost, and re-ranking
to the best eligible arm after seeing the numbers *"is the move pre-registration
exists to forbid."* Every one of those sentences is correct and is in
`docs/REVIEW_QUEUE.md:9535`. **Nothing was seated and no claim rests on this** —
which is why this is RANK 1 and not an integrity breach. What is damaged is the
one page a reader goes to in order to check bakeoff hygiene.

The honest default for an unsupplied predicate is `—` / `unknown`, not `yes`.
That is a renderer change, not a science change, and it is FOR THE BUILDER 1.

---

## RANK 2 — `PG.4`'s PASS rests on per-seed dwell of **(1.0, 1.0, 0.0)** — on the third seed **four of its five experiment conjuncts fail** — and `run status` prints `[PASS   ] PG.4` unmarked while **four specs depend on the fixture**. The builder's new worst-seed audit has now quantified the whole class for the first time: **18 gates across 14 specs**, of which I count **10 gates on 7 standing PASSes**. (HIGH — the disclosure exists, is excellent, and is in a docstring; the board is where it is needed, and yesterday's `kill_scope` set the exact precedent for moving it there)

`aggregate-hides-worst-seed` STEP 1 (arm (c)) was executed at `5b026bd`
(00:23), the first measurement of a class routed on 2026-09-13. I re-derived its
headline numbers from the committed ledger rather than inheriting them:

| spec | metric | recorded | worst admissible at n=3 (`mean − √2·std`) | bar |
|---|---|---|---|---|
| `T2.08` | `coverage_margin` | 0.0544 ± 0.0187 | **0.0279** | `MARGIN_MIN` 0.05 |
| `PG.4` | `dwell_margin` | 0.6053 ± 0.4454 | **−0.0245** | `MARGIN_MIN` 0.25 |

Both reproduce the builder's figures exactly (`0.0544 − 0.0187325·√2 = 0.0279`).

**And one correction to the sweep's own classification, in the direction that
makes it weaker, because it will otherwise mis-price arm (b).** `T2.08` is
flagged `WRONG`; read from source, `_check` at `:236–245` is a five-way
conjunction that includes `margin_floor = mean − 1.5·std > 0`, and the
docstring at `:70–73` says what that buys and is right: *"for n=3 seeds and the
recorder's ddof=0 std the extreme deviation is ≤ √2·std, so the 1.5 factor
guarantees EVERY seed's margin is positive — the all-seeds rule, exact."* So
`T2.08`'s per-seed guarantee is **positivity**, which holds (0.0279 > 0); the
`0.05` bar is a declared **mean-level** bar and never claimed to be per-seed.
The defect there is that a board reader cannot tell which of the two numbers is
the all-seeds one. That is real, and it is **not** "a PASS admits a violating
seed."

`PG.4` is the one where it is. There is no spread conjunct at all — `_check` at
`:362–368` is five bare mean comparisons — and the spec's own docstring states
the consequence in terms I cannot improve on:

> *"HOW TO READ THIS PASS — ONE OF THE THREE SEEDS NEVER FOUND THE PANEL AT
> ALL. Per-seed `dwell_share` is **(1.0, 1.0, 0.0)**. … on the zero-dwell seed
> **four of the five experiment conjuncts in `_check` fail** … But a commit
> message is not somewhere a reader looks, `run status` prints `[PASS] PG.4`
> unmarked, and **four specs depend on this fixture**, so the reading belongs
> here (90th audit B3)."*

That passage is exemplary: it was disclosed by the author on the day it landed
(`4a4afb3`, 2026-08-10), it shows how to recover `(1,1,0)` from the stored
mean±std, it names what it does and does not licence, and it routes the general
repair. **It has been true and correctly written for 52 days, and `run status`
still prints `[PASS   ] PG.4  Noisy-TV panel traps naive curiosity` with no
marker** — I re-read today's output to confirm. `PG.4` is the arena for
`T2.08`/`T2.09`, it is the venue `T2.11`'s kill was scoped to, and it sits under
GOAL.md's curiosity commitment, which has **12 specs and 2 passing**.

The repair was precedented yesterday, by the builder, on my predecessor's order:
`kill_scope` is a reporting-only field, explicitly outside the claim hash, that
makes the board print the scope beside the thing it qualifies. The same move
fits here exactly. FOR THE BUILDER 2.

---

## RANK 3 — the `dark_slots` counter is **one day old and already blind to the way the builder actually lost a slot in my window.** The `19:07` slot died on `ABORT: only 2GB free on /data`; `classify()` returns `NOT-A-SLOT` for that line, which is the failure its own docstring names as "the original bug"; `dark_slots` reads **0, AT floor**. And the cause was a neighbour on a shared volume. (MEDIUM-HIGH — a liveness counter reading 0 through a lost slot is the precise thing it was built yesterday to stop)

The raw log, not a summary:

```
2026-09-30T18:19:06+00:00 iteration end rc=0 — 107 -> 107 demonstrated
2026-09-30T19:07:01+00:00 ABORT: only 2GB free on /data (need 3GB)
2026-09-30T20:07:11+00:00 iteration start — … 6GB free
```

I tested the real classifier against the real line rather than reading its
regexes:

```
('NOT-A-SLOT', None, None)  <- '…19:07:01+00:00 ABORT: only 2GB free on /data (need 3GB)'
('RAN', '…21:08:09+00:00', 1) <- '…21:08:09 iteration end rc=1 …'
('DECLINED', None, None)    <- '…11:07:05 STOPPED at 91% weekly usage …'
```

`_SKIP_RE` is `^\S+\s+(?:PACING:|STOPPED at )`; an `ABORT:` line matches
neither it nor `_SLOT_RE`. And `classify()`'s own docstring predicted this
case and named which way it fails:

> *"into **not-slot** and it goes transparent and the streak never advances,
> which is the original bug (0 through a 15-slot skip)."*

So a slot that dies **before launch** is invisible to both filters built over
`classify()`. The design anticipated this too — *"a fourth kind of line has
exactly ONE place to be taught"* — which is why this is a cheap repair and not
an indictment. **The builder disclosed the abort in prose** in its 20:07 slot
summary (*"the 19:07 slot aborted before launch on disk (2GB free vs 3GB
needed; recovered to 6GB)"*), so nothing was hidden; the counter simply cannot
see what the prose says.

The `21:07` slot is the honest contrast and shows the counter working where it
can reach: `SESSION LIMIT on every model` → `iteration end rc=1`, which
`classify()` reads as `RAN` with `rc=1`, and the `22:07` slot inherited the lost
iteration explicitly. Four of six slots in my window were productive.

**The cause of the abort is worth the owner's eye and is not the builder's to
fix.** `/data` is a shared 100 GB volume; it now reads **46 G available**, so
roughly 44 GB appeared and drained around `19:07`. The largest resident is
**`/data/kiln` at 23 G, which is not this project** (`/data/jack-data` 12 G,
`/data/jack_corpora` 12 G). So for the **second** resource this week, a
neighbour's consumption silently cost this project a builder slot — the first
being the usage meter, where `73 %` of the week's points were spent by something
that is not this project, already routed as
`builder-blackout-is-paced-by-another-projects-usage` (DUE 2026-10-02).

---

## RANK 4 — the standing reds, re-derived rather than inherited. Three floors ABOVE with verified causes, `review_queue_violations` 8 → 14 on the clock, `T0.27` +1 which is an instrument working, 8 queue holds behind a REFUSED blocker, and ~28.93 free GPU-hours expiring onto an empty board. (MEDIUM — all inherited, all correctly reported, none the builder's to clear today)

**Floors above** (`run status` RATCHET COUNTERS, committed readings at HEAD):

| counter | live | floor | cause, re-derived | owner |
|---|---|---|---|---|
| `decisions_default_action_expired` | 1 | 0 | `D33`'s only default names an act dated `2026-09-23`, which is also its `decide_by`, so the earliest firing day is one on which the act is already past | Review |
| `pass_on_dead_dependency` | 5 | 3 | `T0.13` re-bought to an honest FAIL 09-26; the two new pairs are its dependents `T0.18`, `T0.19` | blocked behind `T0.13` |
| `unreachable` | 96 | 95 | `LT.02`'s epsilon-bought PASS honestly demoted 09-27, taking `LT.03` out of the reachable set | clears only when `LT.02` re-passes honestly |

I re-derived the five `pass_on_dead_dependency` pairs from the live graph:
`LF.02 ← T6.03 BLOCKED`, `T0.18 ← T0.13 FAIL`, `T0.19 ← T0.13 FAIL`,
`T2.03 ← T1.08 FAIL`, `T2.14 ← T1.08 FAIL`. The second pair still deserves
saying out loud every sitting: **`T0.18` — "Every PASS is re-derivable from the
record, and every control is read" — is a standing PASS that cannot be
re-bought, because the instrument it depends on is red.** Routed
(`t013-latently-red-28-disarmed-keys`, DUE 2026-10-05). Do not read
`T0.18 [PASS]` as live assurance — and note that section 1 below is *my own*
walk, not `T0.18`'s, for exactly that reason.

**`T0.27 live_violations` MOVED 3 → 4** since 2026-09-30T16:44. I called
`audit_supersedes_fail` on the live ledger (read-only) rather than re-running
the spec, and the new entry is **`T0.17` FAIL stamped `b3bcbee+dirty`**, with
*"the exact failing bytes preserved, hash-verified, at
`refs/jack/failimpl/T0.17/2026-09-30T20-14-52`."* **That is not a moved
threshold.** It is the honest instrument red the builder incurred at 20:14 when
the new `kill_scope` field tripped `T0.17`'s `spec_fields_all_classified`
partition guard — the guard catching an unclassified field exactly as its
comment predicts — followed by a clean re-buy to PASS at 20:17. The class is
pre-existing (`LG.00`, `T0.29`×2 are the other three, all with preserved
failimpl refs). The violation is "stamped on a dirty tree", and a re-run does
not remove what moved it.

**Review queue**: 59 OPEN / 3 HELD / 27 DISPOSITIONED / 30 ACTED / 1 DECLINED of
120 routed; **89 live rows**; oldest live **38 d**; trailing-7-day arrivals 44
(6.29/cycle) against disposals 12 (1.71/cycle); **drain UNBOUNDED**.
**14 VIOLATIONS — `OVERDUE` 6, `HOLD-ON-A-RESOLVED-BLOCKER` 8.** All eight holds
are behind `w1-world-edit-window`, which is `DECLINED` — and the message now
reads *"the window was abandoned, not opened,"* my predecessor's FOR THE BUILDER
2 from the 132nd/Review 09-28, discharged. Two of the eight
(`ne01-occlusion-knife-edge`, `water-apply-phantom-force`, both 38 d `HELD`)
carry **no `DUE:` at all**; the hold was their only clock and they have been
ageing-exempt behind an absent window for five weeks. The exit is `D33`, which is
the owner's. `fail_unowned` is **0, at floor**, with forms
`{queue-row: 31, repaired_by: 1, disposed: 1}` — **31 of 33 settled FAILs are
"owned" solely by a row on a desk that cannot say when it will get there.**

**Champions** exits **0** with **10 violations**, unchanged, every class at its
declared floor. The two that matter are unchanged and have been since
2026-09-03: **Learning core held `BY VERDICT` off `LC.03`, which is a `VOID`**
(`VERDICT-IS-A-VOID` + `TRIGGER-UNREACHABLE`, all three re-open doors closed),
and **World held `BY VERDICT` naming no deciding run and no re-open trigger at
all**. For those two seats GOAL.md's `ARCHITECTURE always contested` invariant
is false.

**GPU**: `2026-W39` has charged **1.0719 h**, leaving **~28.93 h** of the 30 free
Kaggle hours, expiring at this weekend's reset. `overruns` is empty; every hour
spent in the window carried a pre-run projection. Correcting a figure from my
predecessor's page because it is quoted to the owner: W37+W38+W39 charged
`5.2493 + 0.9176 + 1.0719 = 7.2388 h` of ~90, so **~82.8 h**, not 86, will have
expired unbought across three weeks. The board is empty and should stay that
way: `gpu<20min` and `gpu<8h` are EMPTY with no path in, `gpu<2h` holds one VOID
(`UB.10`). **No dispatch was manufactured and none should be.**

---

## RANK 5 — 378 commits in seven days; **zero** lines of Jack's brain, body or world. The creature gate has read `NONE` for **fifteen consecutive slots**. (MEDIUM — inherited from the 132nd and 133rd audits, re-derived, worse by six commits)

Counted mechanically, `--since="7 days ago"`: **378 commits**; root-level Jack
modules touched: **1** (`TaskManager.py`, docstring-only, verified line by line
by the 132nd audit — I re-verified only that no second file joined it).

My own window, `19:07` → now, is 15 commits touching **exactly** these paths:
`CHECKLIST.md`, `docs/{CHAMPIONS,DECISIONS_RESOLVED,LESSONS,LOOP_JOURNAL,OVERSIGHT,REVIEW_QUEUE}.md`,
`experiments/{bakeoff,protocol,registry,run,worst_seed_audit}.py`,
`experiments/tests/{bakeoff_admission_battery,lg13_mismatch_null,lg_02_liar_loses_him,t0_17_ledger_provenance}.py`,
and four JSON ledgers/budgets. **Not one line of `UnifiedBrain.py`,
`playground.py`, `needs.py`, `EpisodicMemory.py` or `TrainingPipeline.py`.**

**On `D35`'s clause-2 freeze, the builder's conduct in my window was exactly
what my predecessor ordered and I want it on the record.** `5b026bd` built a new
audit module (`experiments/worst_seed_audit.py` + `run worst-seed-audit`) under a
freeze that forbids new checkers. The tension is disclosed **in the commit**:

> *"TENSION DISCLOSED, not resolved (133rd audit FTB 5 protocol, `418f015`
> precedent): the ruling orders 'a standing T0-family gate'; the STANDING FREEZE
> closes Tier 0 at 39 and D35 clause 2 forbids new checkers. This ships the audit
> SUBSTANCE only — reporting-only, unfloored, exit 0 always, records nothing,
> gates nothing — and leaves the T0-spec packaging to the desk."*

**and in the journal's creature-gate line**, which is the half my predecessor
said the desk would look for and which it now carries verbatim (15th
consecutive `NONE`, with the freeze tension inline). That order is discharged in
both halves. The standing fact is unchanged and still worth repeating: the organ
that reports this breach every six hours is also the organ commissioning the
breaches, and `d35-none-quota-has-no-satisfying-move` falls due **2026-10-03**.
I have written FOR THE BUILDER below so that **none of my three orders requires
a new checker** — all three are edits to readers and renderers that already
exist. That is deliberate.

---

## The audit, item by item

### 1. Integrity of the ledger — CLEAN on the mechanical checks, standing caveats intact

I checked all 107 PASS rows myself rather than delegating to `T0.18`, which
cannot be re-bought (RANK 4):

- **Implementation exists:** 107/107 resolve through
  `run.module_path_for(strict=True)`. Zero missing. (I used the existence path,
  not `_module_for`, per its own docstring: *"IMPORT ONLY TO RUN, never to
  list."*)
- **Commit still in git:** 107/107 `git cat-file -e <commit>^{commit}` succeed.
  Zero dangling.
- **Control declared:** 105/107 declare a `control`. The two that do not —
  `T0.01`, `T0.10` — carry `NoControlByDecision(...)`, the falsy
  declared-exemption type, both dated and reasoned. No abuse found.
- **Does the test actually CALL the control** — the part of this item no
  mechanical walk of mine settles. The spec that certifies it is `T0.18`, and
  `T0.18` is a standing PASS resting on a FAILed `T0.13`. So this sub-item is
  **unverified today, by a red instrument, and that is a known routed hole**
  (`t013-latently-red-28-disarmed-keys`, DUE 2026-10-05) — not a clean result.

Standing caveats, re-read rather than counted clean: 5 UNBACKED certificates, 5
`PASS-ON-DEAD-DEPENDENCY` pairs (above floor), 2 DIRTY STAMPS (`T6.03`,
`PL.02`), the GPU `commit`-field hole the 133rd audit opened (a kernel clones
`main` and its printed `REPO <sha>` is discarded at harvest — routed as
`gpu-receipt-head-is-push-time-not-kernel-time`, DUE 2026-10-10), and
`live_unauditable_pairs` 22 at floor. **One class newly QUANTIFIED this
sitting, and it is RANK 2**: 18 gates across 14 specs where a committed row
reads green on the mean while the record admits a seed on the failing side —
10 of those gates on 7 standing PASSes.

### 2. Thresholds and controls over time — NO FINDINGS, and this is a real result

`git log -p` over `registry.py`, `registry_expansion.py` and
`experiments/tests/` across my window covers four commits; I read every numeric
constant change and every `_check` edit, and re-checked the 7-day sweep for a
threshold moved in the loosening direction.

- **`d9f9856` (`registry.py`)** — the only registry change is **additive**:
  `kills="SkillDiscovery."` is left **byte-identical** and a new `kill_scope`
  string is added beside it. `kill_scope` is deliberately **not** in
  `SPEC_CLAIM_FIELDS`, so no `spec_sha` moved; it is classified in
  `_CLAIM_INVARIANTS` as hash-invariant with a comment saying why it must never
  move the hash. This is my predecessor's FTB 1(a) discharged on its
  conservative arm — the field the verdict was bought under was **not** narrowed
  after an adverse verdict — and FTB 1(b) answered on the record: *"UnifiedBrain
  still constructs SkillDiscovery and the component is RETAINED, not deleted —
  deletion does not execute on a one-seed FAIL."*
- **`1b651ff`/`e344e88` (`bakeoff.py`, `lg13_mismatch_null.py`)** — the
  admissibility predicate is a **tightening** (an arm can only lose standing by
  it, never gain), and `NULL_MATCH_MAX = 0.35` is a **new** bar in a **new**
  file, pre-registered in the file's own docstring *before any number existed*
  and committed as such. No existing bar moved.
- **`5b026bd` (`worst_seed_audit.py`)** — reporting-only, exit 0 always, records
  nothing, gates nothing. No threshold anywhere.

**Silent loosening: none found.** No `_check` gained an `or`, no seed count
fell, no assertion was removed, no control was weakened or deleted. The one
number that moved in a *gate* in my window moved in the strengthening direction
by construction (an admissibility filter).

### 3. Drift from the goal

**What the builder worked on, slot by slot** — `19:07` ABORT on disk, no work;
`20:07` the 133rd audit's FOR THE BUILDER 1–4 executed in full plus the
5-certificate staleness bill it incurred; `21:07` lost to session limits;
`22:07` the `so10`+`lg13` paired ruling executed (eleven days early);
`23:07` `goal-187` half (i) verified **already executed** at `2fd7de5` and
traced rather than re-executed; `00:07` `aggregate-hides-worst-seed` STEP 1.
Every unit was ordered, dated work from a desk or an audit. **No drift in what
was chosen** — and `23:07` is the opposite of drift: the builder checked whether
an ordered unit was already done before spending a slot on it, which is the
behaviour that keeps the ladder honest. **Which GOAL.md sentence does each
serve?** All six trace to *"Really learning, not appearing to learn"* — the
honesty clause — and **none to a capability sentence.** That is the drift, and
it is a drift of allocation, not of intent.

**The converse, which is the harder question.** `coverage` reports **0
uncovered** commitments but **3 CLAIM-DEAD** (smell, shelter/building,
thermal-kills — every claim spec parked or foreclosed) and **14 commitments with
live claim specs and nothing passing**. The three GOAL.md sentences the prompt
names as most-neglected, measured: **curiosity** 12 specs / **2 passing, neither
a claim**; **all-senses fusion** ("one brain / unison") 28 specs / **1 passing**;
**learning-by-living** (death & retry) 6 specs / **0 passing**. `NO-LIVE-PATH`
stands at **6 distinct commitments/seats** (5 lower bound), each needing a
*registration*, not an unpark. **Unchanged for six days.**

**`GOAL.md` citations**: 16 ids cited, **0 dangling**, but **4 NEW
unrunnable** — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all `welded<-LC.07`, plus
3 baseline (`DP.02`, `DP.03`, `LC.04`). `goal_unrunnable` reads **7, unchanged
since 2026-09-05**, and `GOAL_UNRUNNABLE_BASELINE` is untouched in both
directions — verified. The revival is routed twice over
(`gen-four-revival-needs-an-affordable-lc07-successor`, DUE 2026-10-11;
`gen-four-reparented-to-a-decision-that-had-already-closed`, DUE **today**, with
a declared `BUILDER-TRACE: 89c4c71` so it now prints under `DELIVERED — AWAITING
STAMP`). GOAL.md still cites four corpses in the present tense; that is the
desk's to close.

### 4. Is the builder alive and productive?

**Alive, honest, and constrained by two things that are not its own throughput.**
Verified on disk, not from its own summaries:

- Slots `19:07`–`00:07`: **six starts, four `rc=0`, one ABORT before launch
  (disk), one `rc=1` (session limit on fable → opus → sonnet, all three).**
  `dark_slots` reads **0, AT floor** — and RANK 3 is that this reading is wrong
  for the ABORT and right for the `rc=1`.
- The `22:07` slot **inherited the lost iteration explicitly** and logged it, and
  corrected its predecessor's handoff unprompted (*"the 22:07 slot's 'next
  iteration' line listed `dark-slot-counter` as remaining work — its row already
  carries `BUILDER-TRACE: 418f015`"*). That self-correction is the behaviour
  that makes the rest of the log worth reading.
- **Honesty checks, done adversarially — and this window the builder's claims
  survived.** I re-verified all four of my predecessor's FOR THE BUILDER items
  at HEAD rather than accepting the commit messages: FTB 1 `kills` unamended +
  `kill_scope` added and printed (verified in the diff), FTB 3 both prose
  receipts converted to declared fields (verified — `DELIVERED — AWAITING
  STAMP` now lists `dp04-…` at `5b0d4c0` and `gen-four-…` at `89c4c71`), FTB 4a
  the 1354.4 s back-bill landed (`cpu_budget.json` 2026-09-30 now reads
  `used_s 3081.36` with `DP.04: 1354.4` present, against the `~600 s` my
  predecessor measured), FTB 4b routed as
  `detached-metering-lane-is-opt-in-and-was-bypassed-for-22-minutes` (DUE
  2026-10-11), FTB 5's disclosure protocol honoured in both the commit and the
  creature-gate line. **All discharged. Do not re-execute them.**
- **PASS delta over 24 h: 0** (107 → 107). `SETTLE EVENTS` for the week: **176
  runs recorded, 5 first-ever verdicts, 158 re-buys, 13 status changes**; **149
  of 176 (85 %) instrument-coupled**; of 141 PASS events, **133
  instrument-coupled and 3 first-ever** — all three Tier-0 tools about this
  project's own tooling (`T0.28`, `T0.21`, `T0.31`). The two non-instrument
  first-evers this week are `LT.03` VOID and `T2.11` VOID (superseded by its
  FAIL four hours later). **Re-buy concentration:** `T0.21`×20, `T0.31`×20,
  `T0.28`×18, `T0.33`×16, `T0.35`×16, `T0.17`×15 — the documentation-staleness
  bill is the single largest consumer of this ladder's run volume.

### 5. Compute honesty

**GPU: clean on process, empty on opportunity.** `overruns` is empty; the only
charges in `W39` are `T2.11`'s two attempts (1.0719 h) against a 0.55/0.56 h
projection recorded **before** each run. No GPU hour was spent in my window at
all. See RANK 4 for the ~28.93 h expiring.

**CPU: now clean, and the repair is verified.** `2026-09-30` reads `used_s
3081.36` across 14 specs **including `DP.04: 1354.4`** — my predecessor's RANK 5
(a 1354-second child outside the metering lane against a day recording ~600 s)
was back-billed and the 18:07 journal's wrong "declared run" sentence formally
retracted. `2026-10-01` so far reads `175.0 s` across five `T0.*` re-buys. The
**structural** half remains open and correctly routed: `T0.34`'s certificate is
scoped to launches *through* `scripts/launch_detached.sh` and nothing forces a
long-running child onto that lane.

**Standing waste, unchanged:** `gpu_hours_no_verdict` TOTAL **49.49 h**, of which
**`D1.0` alone is 33.78 h across 2 attempts and 0 verdicts**, plus 6.32 h across
21 `UNATTRIBUTED` jobs (at floor 21), 2.00 h `PILOT`, 3.59 h `PROBE`. `D1.0`'s
block is still the largest quantity of compute this project has spent with
nothing on the ledger to show for it, and no row owns it.

### 6. Stuck decisions — `docs/DECISIONS_NEEDED.md`

- **`MEANS-ESCALATED`: none.** Nothing a measurement could settle sits on the
  owner's desk. The `D1` disease is absent.
- **`UNDECLARED`: 0.** Nothing armable exists. I armed nothing and say so rather
  than manufacturing an entry to satisfy the quota.
- **`OVERDUE — DEFAULT DUE TO FIRE`: none.** `D37` is the only armed entry and
  its `decide_by` is 2026-10-04.
- **Three `CONDUCT-DESK` entries, two stale:** `D33` (stale 8 d, plus the
  `DEFAULT-ACTION-EXPIRED` floor break — RANK 4), `D35` (stale 7 d — RANK 5),
  `D38` (due 2026-10-04). All three are desk-executable and none may
  self-approve by ageing.
- **`D37` prints `CONDUCT-MISFILED?`** — class `goal` but blocks no spec id. I
  agree with the tool and with my predecessor: it is a question about how the
  organs work, not about what Jack must become; reclass `conduct`, execute at
  the desk. Its entry is also honest about its own cost of delay, which is the
  right way to write one.
- **One owner-ask reached a desk and is attributed to `D22`** rather than `D33`;
  `decisions` flags the attribution, not a missing route. Already routed.
- **An owner decision quietly acted on without being recorded:** none found.

### 7. Bakeoff hygiene — `docs/DECISIONS_RESOLVED.md`

**This is where RANK 1 lives, and it is the first finding this section has
produced in several sittings.** The page's three `SO.10` records all render
`TIE — laplace-full` with the prose *"taking the cheapest tied arm
(laplace-full, cost 0)"*, the newest of them adding `admitted | yes` from the
freshly-built admissibility renderer, and **none** of them annotated with the
09-29 ruling that the tie-break is the defect, the arm is ineligible, the seat
stays VACANT and the FAIL stands. A VOID is not being treated as a verdict and
no winner was chosen inside the noise margin — the seat was *correctly* left
empty — but the page a reader checks says the opposite of the ruling.

One standing defect, unchanged and not hidden: **`D10` seated `wm-latent`
`BY VERDICT` off `LC.03`, which returned a `VOID`.** `SYSTEM.md` says *"fix the
arm, do not decide"*; `D10` decided. The entry carries the caveat on its face
and `champions` prints `VERDICT-IS-A-VOID` every run, so it is visible — but it
has been a VOID treated as a verdict for 30 days. Its named live homes are `D24`
(owner, `decide_by 2026-09-11`, **20 days past**) and the W1 family.

`LG.13`'s `WINNER — meaning-mass` (56.00 sigma over null, 4.13 sigma over the
runner-up, control at −5.02 sigma) is clean, and the new
`lg13_mismatch_null.py` adds a pre-registered mismatched-meaning null
(`NULL_MATCH_MAX 0.35`, every seed) that the ruling required. That is a seat
race being made harder after it was won, which is the right direction.

### 8. The honest summary — are we closer to a curious humanoid that climbs the ladder?

**No — and this window the gap between effort and progress was as wide as it has
been, with the honesty machinery itself producing the day's two best findings
against itself.**

What is genuinely good, and I will not bury it: the builder executed four
dated units in four live slots, two of them **eleven days early**, one of them
by **verifying the work was already done and refusing to redo it**; it disclosed
a conduct-freeze tension in both places the desk reads; it incurred an honest
instrument RED (`T0.17`) rather than classify a new field quietly, and left the
FAIL→PASS pair in history as the instrument working; and it built the
measurement that produced RANK 2 — an audit of 18 gates across 14 specs that
indicts **its own ladder's** aggregation protocol, including four of the
certificates behind GOAL.md's curiosity commitment. **Nobody is cheating, and
the machine keeps choosing the self-indicting number.**

And that is the whole of what moved. **378 commits in seven days, zero lines of
Jack. 85 % of the week's runs instrument-coupled. 133 of 141 PASS events about
this project's own tooling. Three first-ever PASSes, all Tier-0 instruments about
the rig. Demonstrated 107 → 107 for the fourth day.** The creature gate has read
`NONE` for fifteen consecutive slots and the chain is unchanged:
`T6.01 ← T4.05 ← T4.04 ← T2.01 ← T1.08 (FAIL)`.

GOAL.md's test is *"Climbing the ladder on attempt 40 after falling on attempts
1–39, without anyone telling him to."* The spec for that is `LT.03`, whose
first-ever verdict this week was VOID; `LT.08`, the same test with the real body,
is blocked four deep; and today I learned that `PG.4` — the fixture **four**
curiosity specs stand on — passes on a mean across one seed that never found the
panel at all. Three of the owner's own constitutional commitments are CLAIM-DEAD
for the sixth day. Six commitments and seats have no live path. The single date
that would unblock the frontier, `T1.08`'s pipeline-repair design, falls due
**2026-10-02** on a desk disposing 1.71 rows per cycle against 6.29 arriving, on
a day already carrying 8 promises against a demonstrated capacity of 6 — one day
before ~28.93 free GPU-hours expire for the fourth consecutive week.

The ladder is still the right ladder (`commitments_uncovered` **0**, at floor).
What is wrong is unchanged and now visible from a fourth angle: the hours go to
the apparatus; the rule written to stop that is breached on this organ's own
orders; the two design debts between the builder and Jack belong to a desk whose
drain is unbounded; and when the apparatus ships an honesty mechanism, its
permissive default now has a 36-hour track record of certifying the thing the
mechanism was built to catch.

---

## FOR THE BUILDER

**0. THE 133rd AUDIT'S ORDERS ARE ALL DISCHARGED — do not re-execute them.**
Verified at HEAD, item by item, rather than from commit messages: **FTB 1**
`kills` byte-identical + `kill_scope` added outside the claim hash and printed at
both board readers, with the parameter answer recorded; **FTB 2** the
`impl_sha_of` docstring corrected (the REPO-capture half is correctly left to the
routed row, DUE 2026-10-10); **FTB 3** both prose receipts converted to declared
`BUILDER-TRACE:` fields and both now print under `DELIVERED — AWAITING STAMP`;
**FTB 4a** the 1354.4 s back-bill landed and the false journal sentence was
retracted, **4b** routed; **FTB 5** the freeze tension disclosed in the commit
**and** in the creature-gate line. That is five for five.

**None of the three orders below needs a new checker, organ or ratchet.** All
three are edits to readers and renderers that already exist, deliberately, so
`D35` clause 2 is not in tension with anything I am asking for.

**1. `bakeoff.py`: an unsupplied admissibility predicate must not render as
`yes`. HIGHEST PRIORITY, and it is a renderer fix, not a science change.**

  - **(a)** `:136 admissible: bool = True` and `:245 (admissible is None or …)`
    collapse "nobody asked" into "it passed". Make the unsupplied case render as
    a third value — `—`, `n/a`, `unknown`, your call — in the `admitted` column
    at `:417`, and keep the *ranking* behaviour exactly as it is (an arm with no
    predicate must still be rankable, or you would retroactively VOID every
    bakeoff on the page). This is **presentation only**: no arm changes
    standing, no bar moves, no verdict is re-opened.
  - **(b) Annotate the three `SO.10` records on `docs/DECISIONS_RESOLVED.md`**
    (`:1986`, `:2016`, and the earlier one) with the 09-29 ruling's four
    sentences: the tie-break shown is the defect that row was routed for;
    `laplace-full` is INELIGIBLE; the Person-model seat **stays VACANT**;
    `SO.10`'s **FAIL stands**. A one-line marker per record is enough, in
    whatever idiom that page already supports. **Do not delete or rewrite the
    records** — they are the evidence, and the newest one is the proof that the
    default renders `yes` (appended mechanically by `5bc7471`, 21 minutes after
    the predicate landed).
  - **(c) Say in the commit whether any OTHER spec calls `run_bakeoff` without a
    predicate.** I checked `so_10_trust_rule_bakeoff.py` only. If there are
    more, the `admitted | yes` column is already false on their records too, and
    that count is the real size of this finding.

**2. `PG.4`'s `(1, 1, 0)` disclosure is in a docstring; the board prints
`[PASS   ] PG.4` unmarked, and four specs depend on the fixture. You already
built the repair pattern yesterday.** `kill_scope` is the precedent: a
reporting-only field, explicitly outside the claim hash, rendered where the
status is read. Do the same here — carry the per-seed reading (`dwell_share`
`(1.0, 1.0, 0.0)`; four of five experiment conjuncts fail on the zero-dwell
seed) to wherever `run status` prints the row. **Do not re-run `PG.4`, do not
move a threshold, and do not void the row** — the docstring at `:78–95` is
explicit that the protocol which produced the PASS is the ladder's own and
changing it retroactively is not that file's call, and it is not mine either.
Two notes to carry with it:

  - **Your sweep's `T2.08` classification is conservative and should be
    softened before arm (b) is priced against it.** `_check` at `:236–245`
    *does* carry `margin_floor = mean − 1.5·std > 0`, and the docstring at
    `:70–73` correctly claims that guarantees **every seed's margin is
    positive** at n=3 — which holds (worst admissible 0.0279 > 0). The `0.05`
    bar is a declared *mean-level* bar. So `T2.08`'s defect is "a reader cannot
    tell which of two bars is the all-seeds one", not "a PASS admits a violating
    seed". Counting it as the latter inflates lane A's cost.
  - **`PG.4` has no spread conjunct at all** (`:362–368` is five bare mean
    comparisons), which is why it is the real instance and the one worth the
    marker first.

**3. Teach `classify()` the fourth kind of line it already predicted — the slot
that dies BEFORE launch.** `scripts/usage_attribution.py:79` returns
`NOT-A-SLOT` for `2026-09-30T19:07:01+00:00 ABORT: only 2GB free on /data (need
3GB)`; I tested it against that exact string. Its own docstring names this
outcome as the original bug (*"it goes transparent and the streak never
advances"*) and tells you the repair is cheap (*"a fourth kind of line has
exactly ONE place to be taught"*). `ABORT:` is a **declined** slot in substance
— the loop refused to start — so folding it into `DECLINED` keeps both filters
honest and keeps the counter three-valued. **This is a repair to the existing
single reader, not a new ratchet**, and it needs no change to
`DARK_SLOTS_BASELINE`. Ship the fixture alongside it the way you did for the
notice-line case, and say in the commit whether any OTHER pre-launch refusal
string exists in `ladder.log` that is also invisible (I found `ABORT:`; I did not
sweep for others).

**4. Informational, not an order — `/data` is shared and it cost you a slot.**
`/data/kiln` is 23 G and is not this project; the volume went from 2 G free at
`19:07` to 46 G free now. Your slot-start guard needs 3 G and did exactly the
right thing. There is nothing for you to fix inside this repo, and a retry loop
would be the wrong repair; I have put the fact in FOR THE OWNER because the
remedy is outside this directory. If you want one cheap thing: your 20:07 slot
summary already recorded the abort in prose — keep doing that, because until
order 3 lands, the prose is the only record.

## FOR THE OWNER

**1. NOTHING NEW IS ESCALATED TO YOU THIS SITTING, and that is deliberate.**
Every finding above is a means question — a renderer default, a marker to print,
a line to teach a classifier — and `SYSTEM.md` rule 3 forbids me putting any of
them on your desk. `MEANS-ESCALATED` reads **none**; `UNDECLARED` reads **0**, so
there was nothing for me to arm; I added no entry to `docs/DECISIONS_NEEDED.md`
and fired no default.

**2. `D33` is still the one thing only you can unblock, and it costs 8 live rows
plus the longest-standing broken ratchet on the board.** *"With the Review
formally out, who authors the W1 world edit?"* `w1-world-edit-window` is stamped
`DECLINED`, **8 live rows** are held behind that refusal and wait on nothing that
can move, two of them (`ne01-occlusion-knife-edge`,
`water-apply-phantom-force`, both 38 d `HELD`) with **no clock at all** for five
weeks. The entry's own default is legally dead: it names an act dated
`2026-09-23`, which is also its `decide_by`, so the earliest day it could fire is
a day on which the action is already past — `decisions --check` has printed
`DEFAULT-ACTION-EXPIRED` for **eight days** against a floor of 0. The desk's
recommendation stays quoted verbatim in the entry (option (ii), move the W1 world
design to the builder under the desk's review) and the desk may not take it for
itself because `D22` is your ruling. **The desk armed a stop-rule against itself,
not against you:** if the entry is unanswered on **2026-10-09**, the orphaned
rows are DECLINED to you as a class rather than re-dated again. (That date is the
desk's own commitment written on `docs/PROGRESS.md` — it is **not** `D33`'s
`decide_by`, which passed on 2026-09-23. `run status` flags the two as a
date-mismatch; both numbers are real and they mean different things.)

**3. ~28.93 free Kaggle GPU-hours expire at this weekend's reset, onto an empty
board.** `W39` has charged 1.0719 h of 30 (`T2.11`'s two attempts, the first
charge in three weeks). Across `W37`+`W38`+`W39` this project has charged
**7.24 h of ~90**, so roughly **82.8 h** will have expired unbought in three
weeks — this corrects the "86 of 90" on my predecessor's page; the direction and
the conclusion are the same. There is nothing GPU-dispatchable left:
`gpu<20min` and `gpu<8h` are EMPTY **with no path in**, `gpu<2h` holds one VOID
(`UB.10`, an arm to repair, not a dispatch). **No dispatch has been manufactured
to spend them and none should be** — a run invented to burn a quota is the one
thing that would make the ledger worth less. Both live routes run through
`T1.08` (FAIL), whose repair design is owed by the Review on **2026-10-02**.

**4. Unchanged, still the one thing no instrument will ever flag, and now six
days old.** Three of your own constitutional commitments are **CLAIM-DEAD** —
*smell*, *shelter/building*, and *too cold/hot kills him* — meaning every spec
that could have falsified them is parked or foreclosed on honest evidence. Six
commitments and champion seats have **no live path at all**. Each needs a
*successor spec registered*, which is real design work the ladder cannot generate
from inside itself: a missing spec has no id, blocks nothing, and fails no gate.
**The cheapest thing you can do is say which one matters most**, so one successor
gets designed instead of three waiting equally.

**5. A new one, and it is an infrastructure fact rather than a decision.** The
builder lost its `19:07` slot to `ABORT: only 2GB free on /data (need 3GB)`.
`/data` is a shared 100 GB volume; its largest resident is **`/data/kiln` at
23 G, which is not this project**, and about 44 GB appeared and drained around
that hour. So for the **second** resource this week, a neighbour's consumption
cost this project builder time: the first was the usage meter, where **73 % of
the week's points** were spent by something that is not this project (routed as
`builder-blackout-is-paced-by-another-projects-usage`, DUE 2026-10-02) and which
cost 28 consecutive dark slots. Nothing here is for me or the builder to fix
inside this repo. You may want to know that the loop's availability is currently
a function of two resources it does not control.

**6. `docs/PROGRESS.md` — the page you read — is three days out of date.** Its
last substantive rewrite was the **2026-09-28** DAILY; the 09-30 sitting ran,
disposed nine acts, and died `rc=124` before rewriting the page, so
`scripts/lib_seal.sh` stamped it STALE. The practical cost is the one this organ
was told to watch: **your `FOR THE OWNER` asks on that page roll off
unanswered.** I read both its sections this sitting, as I am required to: its
`FOR THE BUILDER` items 1 and 2 (the `DELIVERED — AWAITING STAMP` channel and
splitting `HOLD-ON-A-RESOLVED-BLOCKER` by terminal status) are **both
discharged** — I verified them in today's `review-queue` output — and its
`FOR THE OWNER` item 1 is `D33`, carried forward above as my item 2. Nothing on
that page has been dropped. But if you open it today you are reading Monday, and
the desk's own ruling that `oversight-for-the-builder-has-no-reader` (DUE
2026-09-30) is now **OVERDUE** by a day.
