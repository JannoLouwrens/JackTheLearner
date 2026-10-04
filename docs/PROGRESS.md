# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: **FULL (Sunday)**.

**2026-10-04 06:37–07:5x UTC — FULL.** Window: the week.

*The one sentence: **the most valuable thing this desk did this morning was take
a green tick away — `T1.11` had certified since August that the training loss
reaches the actuator, and never once tested that anything CALLS it, which by
measurement nothing outside the ladder does, while the shipped pipeline trains
through the very loss that spec's own control requires to FAIL.***

> **THE PREVIOUS PAGE'S STALE BANNER IS GONE BECAUSE THIS PAGE IS NEW, and the
> thing it was announcing is worth one line before anything else:** this file
> stood unrewritten for 77 hours and its banner understated that by 30, because
> `lib_seal.sh` returns early when a banner already exists. Four consecutive
> audits (131st, 136th, 137th, 138th) ranked that the single most damaging
> reading in the project, and they were right. The cause is structural and is
> routed to the builder below as FTB 5: this page is written LAST by design, so
> the organ that runs out of clock loses the page every time — six consecutive
> `INCOMPLETE` rows in `PROGRESS_LOG.md`, 09-29 through 10-03. **This sitting
> wrote its trend row BEFORE its page** for exactly that reason, which is the
> 138th audit's FTB 6 adopted in advance of its own repair landing.

> This page is the RECEIPT for fourteen commits that already exist, every one
> made before this page was written: `33a41bb` (cross-organ-doc-race ACTED),
> `7d5069e` (xl01 ACTED + successor), `413c566` (w1-cold re-dated), `4fad464`
> (**T1.08's design**), `ebb6792` (d35 re-dated), `d2f4228` (ba03 RULED +
> successor), `897fd38` (this sitting's own two violations repaired in flight),
> `5b9fcfb` (D37's misfile flag disposed), `817c3f9` (T0.31 re-bought),
> `d79053b` (`1^18` steering), `cd36189` (the trend row), `c485304` (D41, D42),
> and the T1.11 pair `(spec, ledger+owner)`. Nothing was held dirty while this
> page was drafted.

---

## THE OVERDUE CLASS, DISPOSED FIRST — `OVERDUE 5 → 0`, `STALE 2 → 0`, `review_queue_violations 14 → 7`

`D28`'s armed default governs and this sitting's first six acts were its
discharge. **Two of the five were FINISHED WORK, and that is the finding about
this desk rather than about the builder.**

| row | disposition | and the honest cause of the break |
|---|---|---|
| `cross-organ-doc-race-voids-certificates` | **ACTED** `b4df9bb` | Fork (c) landed **2026-09-26 12:23**. The row's own note said in terms that *"the ACTED stamp is this desk's"*. **The break is the desk's, not the builder's** — one of eight `DELIVERED — AWAITING STAMP` rows the instrument has printed since the 09-28 FULL ordered that reading. The reading worked; the desk did not read it. |
| `xl01-death-and-retry-has-no-reachable-repair-path` | **ACTED** `b3233a5` | The fork was **dissolved by arithmetic on 09-27**, not decided: both candidate answers measurably false. A desk that had "decided" it would have ordered the expensive route on a refuted premise. |
| `ba03-registered-run-foreclosed-by-d20-class-closure` | **ACTED** `d2f4228` | Limb (ii) **RULED DOWN** — see below. |
| `w1-cold-is-not-lethal-at-night` | **re-dated 2026-10-09** | Fourth break. Its 09-26 re-date promised 09-27 would answer it *"either way"*; the second limb happened — the blocker was `DECLINED` — and **this desk then did not take the simpler question it had promised to take.** |
| `d35-none-quota-has-no-satisfying-move` | **re-dated 2026-10-14** | Genuinely downstream of `T1.08`'s design, which landed 25 minutes earlier in this same sitting. |
| `lt03-icm-trap-not-live-in-flight`, `check-return-type-defect-swept-and-repaired` | **re-armed 10-16 / 10-06** | **STALE: both had NO `DUE:` AT ALL.** Never having one is the same hole as dropping one, with better manners. These were the last two instances in the file. |

**The 7 that remain are all `HOLD-ON-A-RESOLVED-BLOCKER` and the red is
deliberately NOT cleared.** Every one sits behind `w1-world-edit-window`
(`DECLINED`). `w1-cold`'s `BLOCKED-BY:` is **left pointing at the refused window
on purpose**; re-pointing it at a live blocker would clear the violation and
launder the largest structural fact this project has. A clock makes a row honest
about *when*; it does not make the pointer point at anything.

**TWO VIOLATIONS WERE CREATED BY THIS SITTING'S OWN ACTS, caught by the
instrument, and repaired in flight** — `MALFORMED` (a `WAITS-ON` naming the
decision `D33` rather than a row) and `ACTED-WITHOUT-A-COMMIT`. Both refusals
were correct and both conventions they enforced are the right ones. **And the
first of them demoted a Tier-0 certificate:** `T0.31` fell PASS → FAIL at
**06:44:49**, six minutes into this sitting, because its P1 gates live
`MALFORMED == 0` in the real document. Re-bought to PASS at attempt 22 after the
repair, clean salt-1 differential.

## `T1.08` HAS A DESIGN, ON ITS DATE — and Step 0 found the premise rotten

My predecessor re-dated this onto this sitting **as its ordered perishable act**,
ahead of both completeness audits, because ~29 free GPU-hours have expired behind
it for three consecutive weeks. It is delivered.

**STEP 0 (free, source-only, and it must come first): the recipe this noise floor
is measured under is used by NO pipeline.** `t1_08_seed_variance.py:157` says
*"Same recipe as T1.07 and TrainingPipeline. A noise floor measured under a
different configuration would not bound the claims it is supposed to bound."*
The first clause is false, so the second is the finding.

| | `make_action_optimizer` | `TrainingPipeline.py:493` |
|---|---|---|
| optimiser | `Adam` | **`AdamW`** |
| weight decay | none | **1e-4** |
| schedule | warmup → **constant** | none |
| grad clip | 2.0 | **1.0** |

**STEP 1** decomposes the 40.006 % CV with one ~0.3 h job. The seed-dependent
inputs are a **closed set** — task pinned at `manual_seed(900)`, batch order
identical across arms — leaving weight init, stochastic forward, and the one
nobody had priced: the eval sampler at `UnifiedBrain.py:4523` is
`torch.randn(...)` **unseeded off the global RNG**, reached after a full 1500-step
run, so a single seed-dependent draw sits inside a metric whose docstring says it
measures the pipeline. **STEP 2a** (spec-local) or **2b** (recipe-wide:
tail-average/EMA first, then LR decay) follows from Step 1's number.
**Re-buy priced: 0 citing / 19 mechanical / 4 semantic** — `mde_citing` is 0
against `mde_downstream` 49, so **this is the cheapest moment the repair will
ever have.** `MAX_HELDOUT_CV_PCT` 7.0 is named byte-unmoved and **still-FAIL is
pre-registered**, so a reduced number cannot be banked as progress.

## `ba03` — limb (ii) RULED DOWN, against the cheap thing available to me

The row offered this desk a re-label. **Refused, on arithmetic.** `cpu<48h`
over-declares `BA.03` by ~7× (≤172,800 s vs 25,167 s measured); the only class
below is `cpu<2h` = 7,200 s, and `BA.03` measures **8,389 s per seed** — so
moving it there **under-declares by 17 %** and calls the result accurate. The
real defect nobody had named: **there is no CPU class between 2 h and 48 h**
(the GPU ladder has `gpu<20min / 2h / 8h` and no such hole), so all 2–48 h work
must over-declare into the one class `gate_cpu_child` refuses — which is why
`D20` closing that lane foreclosed six ids at once. Routed as
`cpu-class-ladder-has-no-rung-between-2h-and-48h`. **`D20`'s reversal stays
where its firing reserved it: yours.** I could have re-declared `BA.03` this
morning and reported a zero-pass `GOAL.md` commitment unblocked; that would have
been a gate loosened by the desk forbidden to loosen gates, wearing a
measurement as a warrant. **Cost of the refusal, priced out loud: `balance`
stays zero-pass for however long the rung takes.**

---

## Part 1 — the state of progress, in numbers

**Velocity.** 19 runs this week (9 PASS / 10 FAIL) against 11 last week
(5 PASS / 3 FAIL / 3 VOID). **Rework 80.9 %** cumulative (127 of 157 rows at
attempt > 1). VOID 16, BLOCKED 1. `docs/LESSONS.md` stands at 822 entries, 41
commits to it this week.

**Goodhart check — and this is the cleanest reading this check has ever
returned.** Pass rate **42.0 % → 41.6 %** (107 → 106) on an **UNCHANGED registry
of 255.** The registry did not grow; the count fell; and it fell **entirely**
because this desk strengthened a test and demoted it. Rate falling while count
falls and registry holds is neither of the two failure modes this check was built
for. It is the check working.

**The frontier, recomputed rather than quoted.** The `DIRECTION_AUDIT` found 40
specs behind 3 stale results; the live arithmetic is **96 of 255 unreachable**,
and the terminal blockers rank:

| blocker | status | frees alone | blocks in total | impl unchanged |
|---|---|---|---|---|
| **`T1.08`** | FAIL | 3 | **45** | 15 d |
| `NE.01` | FAIL | 7 | 8 | 40 d |
| `UB.10` | VOID | 4 | 5 | 21 d |
| `LT.02` | FAIL | 1 | 8 | 7 d |

**The single most important unblocked spec is `T1.08`, the builder was not
working on it, and it could not have been: the design debt was this desk's and
it was nine days late.** That is now discharged.

**Effort vs goal, and it is the number that should be read before any other in
this report: 249 commits in seven days, of which ZERO touched `UnifiedBrain.py`,
`TrainingPipeline.py` or `playground.py`.** 175 touched `docs/` or `scripts/`;
16 touched spec files. Not one commit this week touched the thing that is
supposed to learn.

**Constitution coherence.** No new contradiction found between `GOAL.md` and
`SYSTEM.md` this week, and nothing in either was edited by this desk. One
tension is worth naming rather than reconciling, because it is the substance of
`D41` below: `GOAL.md` says *"the system is the product, not the model"*, and the
repo currently contains **two** systems — the ladder's rig and
`TrainingPipeline.py` — with all 106 certificates describing the first. That is
not an internal contradiction in the text; it is the text having become true of
something ambiguous.

## Part 1b — the honest paragraph, no numbers

Are we closer to a creature that lives, learns, and is known — or just busier?
Busier, and this week the gap between the two became measurable rather than
arguable. The machinery that watches this project got better again: a test that
had been quietly green since the summer was found to be measuring the wrong half
of its own stated question, and it was taken down rather than defended. The
largest blocker on the board stopped being an acknowledged debt and became a
design with steps, costs and a pre-registered expectation of failure. A desk
refused a repair that was available to it, cheap, and would have let it announce
an unblocked commitment, because the repair was a gate loosened by the one organ
forbidden to loosen gates. Two violations that this sitting itself committed were
caught by an instrument and repaired inside the hour. None of that is small, and
a project that can do all of it is a project whose scoreboard can be believed.

But the creature did not move, and the reason is no longer the builder's meter
or anyone's negligence. It is that the thing we have been building with such
care is the apparatus for judging Jack, and Jack himself has not been edited in
a week. Worse than idleness: we found this morning that two separate bridges
built specifically so that the measuring apparatus and the shipped creature
could not drift apart have both been sitting there uncalled, each of them
proving that the drift they were built to prevent had already happened and that
nothing could see it. The honesty is real. It is honesty about a rig.

**The week's single most important step toward Jack** is the `T1.08` design —
not because it moves him, but because it is the first time the thing standing
between the builder and forty-five specs has had a mechanism attached to it
instead of an acknowledgement. **The most concerning drift away** is that the
loss that trains the module which moves his joints is called by nothing that
would ever run, and we held a passing certificate about it for seven weeks.

---

## Part 2 — the test re-examination

Sampled oldest-passed first: `T0.19`, `T1.02`, `T2.03`, `T2.04`, `TA.02`,
`T3.01`, `T2.19`, `T2.09`, `T2.14`, `T0.18`, `T1.09`, `T1.11`. One yielded a
finding large enough to spend the sitting's remaining clock on, and it is
reported in full rather than alongside eleven thin notes — the other eleven are
re-nominated for next Sunday with their sample position recorded here so the
"least-recently-reconsidered" ordering survives this desk.

### STRENGTHENED

**`T1.11` — "Train/inference path parity" — third conjunct ADDED. PASS → FAIL.**

- **What changed:** `_check` gains `m["shipped_callers"] >= SHIPPED_CALLER_MIN`
  (= 1), fed by a new AST walk over every `*.py` outside `experiments/`.
  `IMPL_DEPS` gains `TrainingPipeline.py`. **Both original conjuncts and both
  original constants are byte-unmoved**; the control is untouched; the new
  conjunct is conjoined, never substituted.
- **Why it is STRONGER, not different:** the spec's own docstring says the
  original defect was that the bridge *"had zero callers in the repo."* It then
  tested the half about reaching the actuator and never the half about being
  called. Measured: **`action_training_loss` has zero call sites outside
  `experiments/`** — the only non-experiments occurrences of the name are its own
  `def` and a docstring mention, which is why the conjunct is an AST walk and not
  a grep. And `TrainingPipeline.py:193` trains through `output['actions']`, which
  **is this spec's own `_control`.** The certificate was bought by proving a loss
  nothing ships reaches the joints.
- **The demotion was pre-registered in the diff before the run**, re-bought
  clean-tree after a first `+dirty` attempt, and **owned in the same act** —
  `t111-certified-loss-has-no-shipped-caller` (OPEN, DUE 2026-10-16) — so
  `fail_unowned` stayed at its floor of 0. Its row pre-refuses both cheap wrong
  repairs: relaxing the new conjunct, and adding a call site on a path nothing
  executes.

**Nothing was weakened.** No threshold moved down, no control softened, no seed
count reduced, no FAILING or VOID spec rewritten to pass.

## Part 2.5 — steering maintenance

1. **PRIORITIES — `1^18` written.** `1^17` was **not stale in content; it was
   unexecuted**, because the builder has been dark since 10-01 — so it is left
   live rather than retired, and `1^18` re-ranks it without discharging
   anything. `T1.08` Steps 0+1 now rank ahead of `1^17`'s W1 registration, on
   perishability alone: that unit is zero-GPU and keeps, this one has a week of
   expiring hours behind it. Step 0 is ordered **free, first, and committed
   alone**, with "stop and route" if it reads differently. No count or status is
   cached in the block; every number points at a living source.
2. **FIELD WATCH — nothing to consume, and that is a real reading rather than a
   skip.** `docs/FIELD_WATCH.md` has not changed since its **2026-09-28**
   `rc=124` draft. `run status` confirms week 9's single finding section is
   already routed (quoted by `t406-latent-floor-was-never-computed`), with 0
   `UNROUTED-FIELD-FINDING`. The untrustable-draft problem itself is owned at
   `field-watch-rc124-page-is-untrustable-and-then-deleted` (DUE 2026-10-13).
3. **SEATS — no finding against any seat this week.** `champions --check` EXIT
   **0** with all 10 violations **AT their declared floors**; `champions_unwinnable`
   4, `champions_trigger_debt` 3, **no seat lost a door in this window.** The
   standing bad news is unchanged and is not re-litigated here: the Learning-core
   seat is still held **BY VERDICT off a VOID** (`LC.03`), with every
   pre-registered re-open trigger a closed door, and the World seat still names
   neither a deciding row nor a rematch trigger.
4. **ORGAN LIVENESS — all four organs fired within cadence; none is silent past
   2× its own.** Builder hourly (last slot 06:07, a PACE-SKIP); overseer 6-hourly
   (06:37 today, plus the completed 12:37 sitting yesterday); field watch weekly
   (09-28, its Monday); this desk Sunday (now). **Silence is never success, and
   there is none.**

### `D30`'s STANDING REPORT — the dark streak beside its perishable cost, in one place

**76 consecutive dark slots / 75.8 hours**, last `rc=0` **2026-10-01T02:17:07**,
against a trigger of 2× the hourly cadence = **2**. The streak is 38× its
trigger. `dark_slots` reads 76 against a declared floor of 0 — **ABOVE floor**,
one of four. The gate is correct and is not being argued with: `week:all models`
83 % at 54 % of the week, line 61 %, and **of this week's 82 shared points the
builder is 16 (19 %), the desks 6 (7 %), and NOT THIS PROJECT 60 (73 %).** First
legal slot ≈ **2026-10-06 14:10 UTC**; the week reset caps the blackout at
≈ **2026-10-07 12:40 UTC** under any meter value.

**And the cost, in the same breath, which is the whole point of `D30`'s
default: `2026-W40` has 0.0 h drawn of 30 free Kaggle GPU-hours, and they expire
Saturday 2026-10-10.** `2026-W39`'s ~28.93 h expired **yesterday** unbought —
the third consecutive week (W37 1.379 h, W38 0.918 h, W39 1.072 h). **~83 free
GPU-hours in three weeks.** No dispatch has been manufactured to spend them and
none should be — but for the first time in three weeks there is now a **designed**
buyer for next week's hours: `T1.08` Step 1, ~0.3 h, DUE 10-08, inside W40 with
two days of margin.

---

## FOR THE BUILDER

**0. THE 135th–138th AUDITS' ORDERS ARE ALL STILL OPEN AND NONE OF IT IS YOUR
FAULT.** Your last slot ended 76 slots before this line and every one of those
reports was written after it. Read the 138th's FTB 1–7 as live and unexecuted. I
am not re-ranking any of it, and nothing below displaces its item 1 (the
steering-size growth fit) which is still the cheapest item on your board.

**1. `T1.08` STEPS 0+1 — your first real unit, and Step 0 is `grep`.** Design in
`docs/REVIEW_QUEUE.md` under `THE DESIGN` on
`t108-pipeline-repair-has-no-design`, `DUE: 2026-10-08` for Steps 0+1 only.
**Verify Step 0 yourself rather than taking it from the design**, commit the
finding alone, and **stop and route** if it reads differently — the whole repair
order depends on it. Step 1 is a diagnostic, not a repair; do not implement 2a or
2b in the same slot, their bills differ by 19 certificates. `MAX_HELDOUT_CV_PCT`
7.0 stays byte-unmoved under every branch.

**2. THE `cpu<8h` RUNG** —
`cpu-class-ladder-has-no-rung-between-2h-and-48h` (DUE 2026-10-15). Add the
class, give it a `child_estimate_s` entry, admit it as a runner child under the
**unchanged** 57,600 s day ceiling and load ceiling, then re-declare `BA.03`
against its **measured** envelope with the measurement's provenance in the
record. `T0.33`'s `cpu_foreclosed == []` must stay green. **Do not** re-declare
the other five `cpu<48h` ids — not one has a measured envelope.

**3. THE ADDITIVE `XL.01` ESTIMATOR** —
`xl01-pooled-conjunct-is-additive-or-it-is-a-loosening` (DUE 2026-10-15). Add
the equal-N pooled ratio as a conjunct **beside** the existing all-3-seeds
per-seed gate; `RATIO_MAX` 0.5 stays byte-unmoved. Report both statistics
whatever the verdict. **Still-FAIL is pre-registered** — a PASS here means
something else changed and must be explained before the certificate is banked.

**4. A LADDER-WIDE READING THAT IS NOT MINE TO IMPLEMENT AND SHOULD NOT DIE AS
PROSE.** From the `XL.01` measurement: **an all-3-of-3 per-seed boolean gate has
power 0.125 at its own bar, at ANY dispersion** — it is not a variance statement,
so no seat count moves it. `NE.08`'s `>= 2 of 3` form lifts that to 0.500 for
free. **Audit the ladder for per-seed boolean conjuncts required on all seeds and
report the count** before any of them is sized by a power pilot. Report only —
**do not convert any spec to `2 of 3`**, which would lower its gate while wearing
a power argument.

**5. THIS PAGE SHOULD NOT BE WRITTEN LAST (138th FTB 6, and I am seconding it
from the inside).** Six consecutive `INCOMPLETE` rows, 09-29 → 10-03, each a
sitting that made real acts and lost only its page. The reason the ordering was
adopted — not holding work dirty — **no longer applies**: `docs/PROGRESS.md` is a
`PROSE_DOCS` member and exempt from the per-spec staleness bill, which this
sitting verified at source. This is a `scripts/review_prompt.md` change and is
yours under the 69th audit's B2 precedent. **Do not remove the `INCOMPLETE`-row
fallback.**

**6. THE `d35` TRIPWIRE CANNOT TELL "NO LEGAL MOVE" FROM "NO SLOT RAN".** It has
read BREACHED for 39 slots and then straight through a 76-slot blackout. A quota
on builder slots cannot be satisfied by a builder that is paced out, and nothing
in its grammar distinguishes the two. Reporting-only repair: print the dark-slot
count beside the breach so the reading is attributable.

---

## FOR THE OWNER

**1. The 90 % hard stop is still yours alone and nobody is watching it — and
`D37` fires tomorrow. Both are already on your desk: `D40` and `D37`.** Read
`D40` first (armed, `decide_by 2026-10-10`, default (v) = the status quo); its
measurements hold and I re-derived the pace arithmetic independently this
sitting. `lib_usage.sh:121` refuses every organ except the regate sweep at 90 %
with *"all agents paused until the owner resumes"*, and resuming needs a
`.usage-resumed` file written **by you**; there is none on disk. `week:all
models` reads 83 %. **Separately and more urgently: `D37`'s default fires
2026-10-05** — monotone, legal, and I am not asking for it to be delayed. But I
disposed its `CONDUCT-MISFILED?` flag this morning by **refusing** the reclass
that would have let a desk execute it, so the firing now stands unopposed, and
its own price is *"the project keeps a `mandatory` guard it has never once been
able to run."* One day's notice is all this item is for.

**2. NO-DECISION: `D30`'s standing report, delivered here as its armed default
requires, with nothing to rule on.** Builder dark **76 slots / 75.8 h**, longest
on record; first legal slot ≈2026-10-06 14:10 UTC; 73 % of this week's shared
pool is another tenant's. **`2026-W40`: 0.0 h drawn of 30 free Kaggle
GPU-hours, expiring Saturday 2026-10-10;** W39's ~28.93 h expired yesterday
unbought, the third consecutive week, ~83 h in three. For the first time there
is a **designed** buyer for next week's hours — `T1.08` Step 1, ~0.3 h, DUE
10-08. All four organs fired within cadence. `demonstrated` 107 → **106**, and
the fall is this desk's own strengthening, not a regression.

**3. Nothing new is asked on `D33`; this is a pointer, not a re-ask. See `D33`.**
Its `decide_by` was **2026-09-23** and is now 11 days past; it is the sole cause
of the one broken ratchet class on your register
(`decisions_default_action_expired` 1 against floor 0), and the 10-02 Review
established its default is **MOOT rather than merely expired** — its object went
terminal when `w1-world-edit-window` was stamped `DECLINED`, so **no desk can
clear this by firing anything.** A separate date, **2026-10-09**, is this desk's
own armed stop-rule against itself and not `D33`'s deadline: on it, the orphaned
rows are DECLINED to you as a class. **`w1-cold-is-not-lethal-at-night` joined
that class this morning rather than taking a fifth invented date**, and its
stop-rule is now unconditional. Its recommendation stays quoted verbatim in the
entry, unchanged.

**4. Two separate bridges built so the ladder and the shipped system "cannot
drift apart again" both have ZERO pipeline callers, and 249 commits this week
touched Jack zero times. Which artefact is the thing we are building? Routed as
`D41`** (class `goal`, default (iii) NEITHER — hold both and repair each instance
on its own row, `decide_by 2026-10-18`), with the recommendation quoted verbatim
there. The short form: `T1.11`'s certified loss has no shipped caller;
`make_action_optimizer` has four spec callers and no pipeline caller; and all 106
certificates describe the ladder's configuration. If `TrainingPipeline.py` is
what would train a living Jack, the honest `demonstrated` count for the shipped
artefact is **unknown**, not 106. **My recommendation is (ii): declare the
ladder's rig the system of record and put `TrainingPipeline.py` on the ablation
block** — deliberately NOT the cheap repair of making the pipeline call the
certified loss, which would turn `T1.11` green again while leaving two
half-certified artefacts in the repo, which is how both drifts happened.

**5. Three of your own constitutional commitments are CLAIM-DEAD and each needs
a successor spec the ladder cannot generate from inside itself. Which ONE matters
most? Routed as `D42`** (class `goal`, default (iv) HOLD — no successor is
designed and the three stay claim-dead and said so, `decide_by 2026-10-18`), with
the recommendation quoted verbatim there. **This is the fourth consecutive report
to raise it and the first to put it somewhere it can be answered** — that is the
`VANISHED-OWNER-ASK` scar being honoured rather than described. `claim_dead` has
read 3 for eight days: **smell**, **shelter/building**, **thermal ("too cold
kills him")**. Every park was right on its evidence; leaving the commitment
claim-dead is the bug, and no instrument can ever ask for the fix because a
missing spec has no id, blocks nothing and fails no gate. **My recommendation is
`thermal`**, on the arithmetic that it is load-bearing for shelter (which has no
consequence without a lethal cold to shelter from) and is the only one of the
three whose successor needs no world edit.

**6. NO-DECISION: what this sitting did not do, named rather than omitted.** The
**ANATOMY and COMPLETENESS audits are deferred a week**, under this desk's own
committed 10-03 ruling that when a sitting cannot hold everything the perishable
item goes first — ~29 GPU-hours expire every Saturday behind `T1.08` and nothing
expires behind an audit. What the cheap readings do show is that the 2026-08-09
scar list has **materially moved**: **voice 0 → 3 specs with 2 passing**, **taste
0 → 3 with 1 passing**, **smell 0 → 2 but CLAIM-DEAD**, **body schema 0 → 1 and
dead on `UB.14`**; `commitments_uncovered` is 0 and `champions` carries explicit
seats for both Smell and Body schema. The half still unaudited is the **cognitive**
one — attention, working memory, imagination, self-model, theory of mind,
teaching — and it is owned with a clock at
`completeness-audit-2026-09-13-the-cognitive-half-is-the-hole` (OPEN, DUE
2026-10-11), so it ages in public rather than inside this paragraph.
**And one disclosure about my own conduct:** I added ~1,960 bytes to
`scripts/ladder_prompt.md`, which has 3,026 bytes of headroom against its 125,000
ceiling, and I trimmed my own block when I saw the number. The organ warning the
builder about that ceiling should not be the one crowding it.
