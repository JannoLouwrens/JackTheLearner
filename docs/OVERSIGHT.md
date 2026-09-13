# OVERSIGHT.md — the overseer's current-state report

> Rewritten every audit. Current state, not a log. The running history is the
> git log of this file.
>
> **91st audit — 2026-09-13, 00:37–01:0x UTC.** Read at HEAD `5dd372f`. The
> builder is awake and running concurrently (`PL.02` attempt 2 detached, pid
> 3752179, started 00:23, verified alive in `ps`). **36 commits since it woke at
> 17:07 yesterday**, across 6 iterations, all `rc=0`. Numbers below are a
> snapshot of a moving tree.

## VERDICT: ON TRACK

The blackout is over and the builder came back better than its guardrails. In
7.5 hours it re-aimed one gate, tightened a second, committed a third,
foreclosed a dead spec, fired the last of the overdue armed defaults, delivered
an ordered probe two days early, and **twice refused a GPU dispatch it was
formally entitled to make**. Sections 1 and 2 are clean. `demonstrated` is
108/245 for the sixth day, and that is the honest cost of a week spent repairing
gates rather than climbing rungs.

**My three findings are all of one kind, and it is not misconduct — it is that
the things holding this system straight right now are sentences, not
mechanisms.** The builder is currently obeying constraints that nothing
enforces, counting classes that no ratchet guards, and working to dates whose
stated justification has expired. Every one of those held today because a
careful organ read its own journal. None of them would survive an organ that
did not.

---

## RANK 1 — the project's largest GPU spend is guarded by prose alone: `dispatch.sh` is 61 lines and checks nothing but "is HEAD pushed" and "is the lock free"

**The standing number.** `D1.0` has consumed **33.78 GPU-hours across two
attempts for exactly one ledger row, and that row is `VOID`** (`ledger.json`:
`D1.0 VOID 2026-09-07T01:57:12`, commit `3a4ccfd` — the only `D1.0` row that
exists). Attempt 1 ≈ 16.17 h in `2026-W35`, attempt 2 ≈ 17.61 h in `2026-W36`,
both reconstructed from `gpu_budget.json`'s per-job records.

**What is queued.** `2026-W37` opened with **30.0 h and 0 spent** (no `2026-W37`
key in `gpu_budget.json`). `D1.0` attempt 3 measures ~17.6 h — **59% of the
week's entire allocation.** And the builder's own replay of attempt 2's verbatim
row through the newly committed successor gate (`7cb00ea`, transcribed to
`REVIEW_QUEUE.md:3617`) says it clears `G0`, the control and **both** learning
conjuncts on all four arms — and then lands on **`VOID (SPLIT-PENDING)`**:
`aprime` leads `d_mlp` by 3.37σ on eval mean (506.4 vs 415.0) while `d_mlp`'s
final-third *training* reward is **higher** (5.411 vs 5.303) with a positive
slope. In the builder's own words: *"a re-run at the same `STEP_TARGET` (750,000)
is likelier to return SPLIT-PENDING than a winner."*

**So the third attempt would take this row to ~51.4 GPU-hours for zero
verdicts** — and the row itself already names that as the thing it exists to
prevent: *"three attempts and 50 GPU-hours for zero verdicts is the pattern this
row exists to stop."*

**Now the finding, which is not about the builder.** I went looking for what
would actually stop it. There is nothing:

```
scripts/dispatch.sh          61 lines total
  line 26   REFUSING: HEAD is not pushed and the GPU VM clones from GitHub.
  line 40   REFUSING: $GPULOCK is held
  — and that is the complete set of refusals.
```

- **No budget check.** `grep -n "remaining\|Budget\|gpu_budget"` over
  `dispatch.sh` and `launch_detached.sh` returns **nothing**. A dispatch that
  would overrun the 30 h week is not refused; it is simply charged.
- **No precondition check.** `_GATES_FROZEN = True` in
  `d1_0_control_path_bakeoff.py:389`, so `run()` does not refuse either. The
  provisional-gate refusal that guarded this spec in August has been correctly
  discharged and nothing replaced it.
- **No authorisation check.** The row's *"an unchanged re-dispatch stays
  forbidden"* and *"this is a dispatch question and it is the Review's"* exist
  in `REVIEW_QUEUE.md` and `LOOP_JOURNAL.md`. No code reads either.

**The builder has refused twice in the last eight hours — 21:54 and 00:27 —
voluntarily, and wrote its reasoning both times.** That is the system working,
and it is exactly why this is a RANK 1 finding rather than a complaint: *the
only functioning guard on the single most expensive irreversible act this
project can take is one organ's habit of reading its own notes.* The loop fires
hourly. The Review's next sitting is today and **the `D1.0` row is dated
`2026-09-14`, so it is not even on today's docket.** That is roughly 30 builder
slots between now and the authorisation, each one reading a journal line that
says the preconditions are *"MET"* and *"satisfied in form."*

**The repair is cheap and it is not a new policy** — it is making an existing
written rule executable. Routed as **B1**.

---

## RANK 2 — `champions.py` prints a class it does not count, and that class grew yesterday: four architectural seats are now permanently unwinnable and no ratchet moved

The overseer's own standing instruction says it plainly: *"The ratchet counts
every class on purpose. Three instruments here shipped counting one —
`coverage.py`, `decisions.py`'s `NO-DEFAULT`, `champions.py`'s `ARENA-MISSING` —
and each paid a 'repair' that lowered its own number."* **This is the fourth
instance, in the third of those same three tools.**

`champions.py:1843` computes a list and prints it:

```python
unwinnable = [s["seat"] for s in seats
              if s.get("arena_pending_dead") and not s.get("arena_welded")]
...
"...and seats no one can ever WIN — every pending arena member welded, but no
 unearned holder to indict (out of the ratchet by scope, not oversight)"
```

**It is computed in the render path only.** `--check` asserts on
`BASELINE_UNDECLARED`, `BASELINE_UNFALSIFIABLE`, `BASELINE_UNCONTESTABLE`,
`BASELINE_ARENA_MISSING`, `BASELINE_VERDICT_UNVERIFIED`, `BASELINE_KINDLESS` and
`BASELINE_TRIGGER_UNREACHABLE` — **and never on `unwinnable`. It is the only
class in the tool with no baseline at all.** `ARENA-UNREACHABLE`, which *is*
ratcheted through `BASELINE_UNCONTESTABLE`, requires `arena_welded` — a holder
to indict. A seat with no holder falls straight through.

**It grew yesterday.** `LG.03` went `VOID-FORECLOSED` (`2a39208`); `LG.04`,
`LG.05` and `LG.06` all `depends_on` it and are the entire declared arena of the
**Language grounding (word → lived skill)** seat (`CHAMPIONS.md:320`). The
builder verified and stated the consequence honestly in its own commit message:
*"ratchet counters — NONE moved, in either tool."* It was right. **That is the
defect.** The list now reads **4**:

    Episodic retrieval · Language grounding (word → lived skill)
    Smell (olfaction)   · Body schema (the model of his own body)

`SYSTEM.md`'s invariant is that **ARCHITECTURE is always contested**. A seat
whose every arena member is welded is not contested, and whether anyone happens
to be sitting in it is beside that point. This class can grow to every seat in
the file without one red number.

**And it is worse than it looks, because two instruments are watching the same
rot and neither sees the whole of it.** `coverage` exits `rc=2` on **4
CLAIM-DEAD** commitments; `champions` prints **4 unwinnable** seats. They
overlap in exactly **one** member — *smell*. The union is **7 distinct
constitutional commitments or architectural seats with no live path**, and no
number anywhere in this repo prints it.

| `coverage` CLAIM-DEAD | `champions` unwinnable |
|---|---|
| smell | **smell** |
| balance | Episodic retrieval |
| shelter/building | Language grounding |
| thermal (kills) | Body schema |

Routed as **B2** (ratchet the class, baseline 4, shrink-only — the standard
idiom) and **B3** (print the union).

**Credit where it is due:** the 90th audit found this list, ran the
counterfactual, and recorded *"no ratchet counter moves"* against its own
expectation. It treated that as a scoping fact. I am calling it the finding.

---

## RANK 3 — 14 rows are due TODAY against a measured capacity of 6, and the premise that put several of them there has expired

`review-queue` reports **0 violations** — every date is still in the future by
hours — and then prints the pile:

```
2026-09-13   14  ##############  !! AMBER: pile
   ...against a measured capacity of 6/cycle
```

**8 of those 14 were dated onto a day that already carried its capacity** when
they were routed (the tool's own `DATED ONTO A FULL DAY` list). The desk's drain
reads **UNBOUNDED** — 40 live rows, arrivals exceeding disposals by 2 over the
trailing 7 days, *"the backlog has no projected end."* The Review has said this
for a week and I am not re-litigating it.

**What is new is that a stated premise behind several of these dates is now
false.** `REVIEW_QUEUE.md:2576` dates `reparenting-the-welded-fifteen` and
`goal-cites-four-specs-that-resolve-to-corpses` to 09-15 with the reason spelled
out: *"Why 09-15 and not today: the input is builder work and **the builder is
measurably switched off**."* The builder has since produced 36 commits in 7.5
hours. That justification has expired and the rows should be re-read in that
light — not necessarily re-dated, but re-read. Flagged to the Review as **B4**;
the date is the Review's to set, never mine.

**Also live and uncounted here:** `coverage` reports **4 NEW unrunnable
citations** — `GEN.02`, `GEN.03`, `GEN.06`, `GEN.09`, all welded behind `LC.07`
— where `GOAL.md` cites specs in the present tense that resolve to corpses. This
is correctly routed (`goal-cites-four-specs`, DUE 09-15, bundled) and correctly
*not* baselined away. No action needed beyond the date question above.

---

## The audit, section by section

### 1. Integrity of the ledger — NO FINDINGS

The 90th audit checked all 108 PASS rows mechanically less than seven hours ago
— commits resolve, controls declared, 106 of 108 controls run with the two
exceptions carrying `"NONE, BY DECISION (52nd audit B5)"`. **No PASS row was
added or altered since** (`demonstrated` 108 → 108 across all 6 iterations), so
that result stands unchanged and I did not spend the budget re-running it.

One ledger fact worth stating plainly: **`D1.0` has exactly one row and it is
`VOID`.** Attempt 1's row was superseded under the T2.02 precedent, correctly.

### 2. Thresholds and controls, over 7 days — NO FINDINGS, and the direction is *tightening*

Every numeric constant that moved in `experiments/tests/` and
`experiments/registry*.py` over 7 days, with its direction:

| commit | change | direction |
|---|---|---|
| `1bd42dc` | `PLANNER_CALIB_MIN = 1.0` **added** as a VOID conjunct, checked *before* `blind_calib_rate` | **tightening** (and it cost — `LG.03` is now foreclosed) |
| `4f2a1a6` | `MIN_DISTRACTOR_EVAL` 9 → 12 | **tightening** |
| `021d302`, `fa88f05` | `N_PROPERTIES` 15 → 16 → 17 | **tightening** (more properties asserted) |
| `7cb00ea` | `TWIN_DRIFT_SIGMA = 3.0`, `LEARN_MARGIN` derived from a frozen probe table | new conjunct, derived not fitted |
| `a4132c8` | `EYE_RADIUS_R2_MIN` **unmoved at 0.80**; referent changed `r2_ua` → `r2_raw_pixel` | see below |
| `8f2990d` | `MIN_LEARN_SIGMA = 3.0` → `3.0` with comment *"UNMOVED"* | no move |

**Zero thresholds moved in the loosening direction. Zero controls deleted or
weakened. No `_check` gained an `or`. No seed count reduced.**

**I looked hardest at `a4132c8`, because a gate whose *referent* changes while
its *bar* holds is the perfect disguise for a loosening** — and the effect was
to take `PL.02`'s eye gate from VOID (`r2_ua` −0.0017) to CLEARING
(`r2_raw_pixel` 0.9292). It survives scrutiny:

- It executes a **Review ruling** (`5e39771`, 09-11), not a builder preference.
- The ground is algebraic and checkable: `r2_ua` is the **subtrahend** of the
  claim's own effect size `R_pl = r2_pl − r2_ua`, so gating on it capped the
  maximum reportable gain at ≤0.20 against an observed 0.94. A guard that caps
  the thing it was added to protect is broken in the direction of *nonsense*,
  not of leniency.
- The builder proved the new branch **fires** rather than merely passes — VOIDs
  at grey@64 (0.5614), grey@96 (0.6861) and one ulp under the bar — and checked
  domination symbolically.
- **And the re-aimed gate did not rescue the spec.** `PL.02` attempt 1 harvested
  **VOID** anyway, on the `learn_ok` conjunct (0.667). A loosening that leaves
  the run VOID is not a loosening.

The same logic was independently confirmed from the other direction hours later:
the `SM.03` F2 probe found a venue readable at 99.17% behind a bearing-blind
readout, producing the lesson *an alive-proof built from the same readout as the
arm it validates cannot tell a dead venue from a dead instrument.* `PL.02`'s
re-aim is that lesson applied. **Clean.**

### 3. Drift from the goal — NO DRIFT

Every unit of the last 24 h traces to a `GOAL.md` sentence:

| work | `GOAL.md` sentence |
|---|---|
| `PL.02` gate re-aim, dispatch, VOID harvest | *"PLASTIC ONLY — nothing inside him is frozen"* — `PL.02` is the **sole registered falsifier** of that decree |
| `LG.03` tightening + foreclosure | *"he learns words the way every child does"* |
| `SM.03` F2 probe | *"SMELL… the sense that works when sight fails"* |
| `D1.0` successor gate | *"one interconnected brain"* — largest unblock in the project (frees 35 / blocks 38) |
| 5 armed defaults fired | *"protects the honesty of watching what happens"* |

**The converse, which is the harder question and the worse answer.**
One-brain/unison: **1 passing spec of 27.** Curiosity: 2 of 12, and its only
headline fixture (`PG.4`) is the one whose trap demonstrably did not fire on a
third of its seeds. Sleep: 0 of 5. Fast/slow: 0 of 8. Four constitutional
commitments are claim-dead. The fresh-dispatch queue is **0 deep** — `coverage`
reports 6 dispatchable specs *"of which 6 VOID → only 0 is a FRESH dispatch."*

### 4. Is the builder alive and productive? — ALIVE, and the most productive night in a week

6 iterations 17:07 → 00:27, **all `rc=0`**, 36 commits, zero identical repeated
failures, no pause, no load abort. It walked itself off Fable (`week:Fable`
pinned 100%) onto Opus via `D14` option (b), correctly, every slot.

**PASS delta: 0.** Under `SYSTEM.md` (*"any session that makes the machine
better at catching its own errors has done the whole job even if no spec
passed"*) this was a full night's work, and it earned that reading: it caught a
false premise in a ruling it was executing, a receipt that could never clear, a
control unreachable by algebra, and a `T0.13` blindness — each one found by
reading code rather than by any tool.

**Meter:** `week:all models` **78%** against ~83% week-elapsed — under the line,
builder released. Resets 09-14T05:00. `week:Fable` 100%.

### 5. Compute honesty — the W36 refusal was CORRECT; the standing number is the problem

**`2026-W36` closed at 17.72 h of 30, leaving 12.28 h unspent, and that was the
right call** — attempt 3 measures 17.61 h, did not fit, and the builder refused
to scrape it out under direct pressure from two desks. **`2026-W37` is open at
0 of 30.**

No GPU hour in the last 7 days is unaccounted: every `charged_jobs` entry in
`W35`/`W36` maps to a `D1.0` attempt, and both attempts produced an honest
recorded result. **The problem is not waste in the dishonest sense — it is
yield: 33.78 GPU-hours, one ledger row, `VOID`.** See RANK 1.

CPU is healthy: 1,821 s of 57,600 used today, correctly billed per-spec.

### 6. Stuck decisions — CLEAN, and one is due TODAY

`decisions --check` EXIT 0. **3 armed, 0 overdue, 0 `MEANS-ESCALATED`, 0
`UNDECLARED`, 0 `UNROUTED-OWNER-ASK`, 0 `VANISHED-OWNER-ASK`.** The overdue
queue was emptied yesterday between 17:09 and 17:25 — first time since 09-08.

**I have no `UNDECLARED` entry to arm this audit, because there are none.** The
standing instruction to arm at least one per audit is satisfied vacuously and I
say so rather than manufacturing one.

- **`D25` — `decide_by` is TODAY, 2026-09-13.** Fires tomorrow if unanswered.
- **`D19` — `decide_by` 2026-09-14, costs 3 specs**, blocks `HR.1`–`HR.4` and
  holds `cpu<10min` empty.
- `D20` — `decide_by` 2026-09-18, costs 0.

**Nothing was quietly acted on.** I checked the five firings of 09-12 against
`DECISIONS_RESOLVED.md`: each carries the required wording and a reversal path.
`D26` fired option **(iv) MEASURE ONLY** rather than the **(i)** both desks
recommended, *because a default may not loosen a gate* — the safety clause
working against the builder's own interest, unsupervised. That is the single
most reassuring thing in this audit.

### 7. Bakeoff hygiene — NO FINDINGS

No VOID was treated as a verdict. `D1.0`'s VOID is being treated as a VOID —
that is the whole of RANK 1's tension. No winner was chosen inside a noise
margin; `SPLIT-PENDING` exists precisely to refuse that and it fired. `LG.03`'s
foreclosure rests on both means being under both bars in a **deterministic** run
(33 shared metrics across two attempts, zero differing), verified independently
by the builder *before* stamping, and explicitly **not** on the refuted
teacher-caps-twin mechanism.

### 8. The honest summary

**Closer — but on the instruments, not on the creature, and for the sixth day
running.**

Jack cannot do one thing today he could not do on Monday. `demonstrated` has
read 108/245 for six days. Four of his owner's constitutional commitments have
no living claim; smell, balance, shelter and "too cold kills him" are each
behind a spec that was honestly parked and never succeeded. He has one passing
spec out of twenty-seven for the unified brain that is the whole thesis. The
fresh-dispatch queue is empty — not because the loop is idle, but because **the
ladder has run out of rungs that can be climbed without a redesign.**

What genuinely improved is the project's ability to distrust itself. In eight
hours the builder found that a ruling it was implementing rested on a swapped
pair of seed labels, that a receipt it depends on could never clear, that a
control it had just written was unreachable by algebra, and that `T0.13` — the
detector whose entire job is finding gates that cannot fire — is blind to the
whole class. Four blind spots, none of which any tool here would have surfaced,
all found by reading source. That is real and it compounds.

But the thing I keep circling is the shape of my own three findings, and I did
not go looking for it — it fell out. **A dispatch that would spend 59% of a
week's GPU on a predicted null is stopped by a sentence. A class of permanently
dead architecture is watched by a print statement. Rows are dated against a
premise that stopped being true yesterday afternoon.** Every one of those held
today, and every one held for the same reason: an unusually conscientious organ
read its own notes and chose correctly. That is not nothing — it is, genuinely,
the culture working. It is also the exact failure mode `GOAL.md` warns about one
level up: *"A loss curve is not learning. A README saying 'Working' is not
learning."* A guardrail that only works when the thing it guards is already
behaving is not a guardrail. It is a README.

The builder is currently better than its own rules. The repair is to make the
rules as good as the builder, while it is still true that they are not.

---

## FOR THE BUILDER

**B1 — RANK 1, and do this before any GPU dispatch this week. Make
`dispatch.sh`'s existing written rules executable.** Three refusals, all of
them encoding a rule that is *already binding in prose*, none of them new
policy:

1. **Budget refusal.** Before submitting, read `gpu_budget.json` for the live
   week and refuse if `projected_hours > remaining()`. Use
   `Budget.remaining_range()`'s floor, per the `opening_balances` note. Print
   the arithmetic in the refusal. This is the rule the builder followed by hand
   on 09-12 when it declined to scrape `W36`; encode it.
2. **Authorisation refusal for a re-dispatch.** If the spec already has a
   settled row and nothing in its `IMPL_DEPS` has changed since that row's
   `commit`, refuse with `REFUSING: unchanged re-dispatch` and name the row.
   `REVIEW_QUEUE.md` already calls this forbidden; today nothing reads it.
3. **State the projection.** Refuse unless the caller passes an explicit
   `--projected-hours`, and record it beside the job in `gpu_budget.json` so
   projected-vs-actual becomes auditable. `D1.0`'s 17.61 h estimate is
   currently only in prose.

   **Red-first, as always:** show each refusal FIRING against a constructed
   case before you show it passing. `D1.0` attempt 3 at 17.6 h against `W37`'s
   30 h must *not* trip (1); it must trip (2) unless the gate commit `7cb00ea`
   counts as a change to `IMPL_DEPS` — decide which and say so in the
   docstring, because that judgement is the whole substance of the guard.

   **This does not decide the `D1.0` dispatch question.** That is the Review's,
   on its own row, DUE 09-14. Do not pre-empt it and do not dispatch attempt 3
   before it rules.

**B2 — RANK 2. Ratchet `champions.py`'s `unwinnable` class.** Add
`BASELINE_UNWINNABLE = 4` beside the other baselines and assert it in `--check`,
shrink-only, in the same idiom as `BASELINE_UNCONTESTABLE`. Today's four are
`Episodic retrieval`, `Language grounding (word → lived skill)`, `Smell
(olfaction)`, `Body schema`. Record in the comment *why* it shrinks — by
registering a runnable spec or re-parenting an arena member off its foreclosed
root — and *never* by deleting a seat or an arena reference, exactly as the
neighbouring baselines say. Note in the same comment that `LG.03`'s foreclosure
(`2a39208`) is what took it to 4 and that **no counter moved at the time**;
that is the fact the baseline exists to stop recurring.

Per the 90th audit's own lesson, **run the counterfactual rather than reasoning
about the predicate**: confirm read-only that the new assertion is green at 4
today and red at 5.

**B3 — the union nobody prints.** `coverage`'s 4 CLAIM-DEAD and `champions`'
4 unwinnable overlap in exactly one member (*smell*); the union is 7. Print that
union in one place — a single line in whichever tool you judge the better home,
naming each commitment/seat and which instrument sees it. One number for "how
much of `GOAL.md` currently has no live path" does not exist today, and it is
the number section 8 of every audit is really trying to answer.

**B4 — carry to the Review, do not act on it yourself.** `REVIEW_QUEUE.md:2576`
dates two rows to 09-15 with the explicit reason *"the builder is measurably
switched off."* You are not. Flag the expired premise on the rows so the Review
reads it this morning. **Do not re-date anything** — dates on that file are the
Review's.

**B5 — `T0.13`, when the 09-20 sweep authorises it, not before.** Your own
measurement (`484e090`) is the strongest evidence in the file: perturbing
`D1.0`'s control key to each of `0, 1, −1, ±1e9` returns `moved=False` five
times of five, because `t0_13_gates_are_live.py:403` compares
`("STATUS", out.value)` and every branch of that `_check` returns VOID. The
detector for dead gates cannot see a dead gate in any VOID-heavy rig. You were
right to leave a passing certificate alone; make sure the sweep row carries the
five-of-five receipt so the decision is made on the number.

---

## FOR THE OWNER

**1. `D25` is due TODAY (2026-09-13) and its default fires tomorrow.** If you do
nothing, option **(iii) FIX THE SEAL** fires: `lib_seal.sh` learns to read a
dying run's own committed acts, so a Sunday FULL that committed its whole page
is no longer banner-ed *"THIS IS A DRAFT, NOT A FINDING … UNVERIFIED"*
identically to one that committed nothing. **I have no objection and recommend
letting it fire.** It is the only legal default of the three — it moves no
threshold, spends nothing, and is monotone. The cost is stated honestly in the
entry: Sunday FULLs keep exiting `rc=124` and keep looking unhealthy to anything
reading exit codes alone.

**2. `D19` is due 2026-09-14 and it costs 3 specs** — `HR.1`–`HR.4`, the whole
hearing-claim family, and it is why `coverage`'s `cpu<10min` class reads EMPTY.
Its default is **NO FETCH** (no corpora downloaded outside the repo), which
leaves hearing's claims *"runnable-on-paper and blocked-on-disk"* and leaves a
visible red rather than a quiet workaround. That is the correct default and I am
not asking you to change it — but you should know that letting it fire keeps a
constitutional sense unbought, and that the honest alternative is a spec
amendment through the strengthen-only lane, not a default.

**3. NO-DECISION — the GPU question, stated because the money is yours.**
`D1.0` has spent **33.78 GPU-hours across two attempts for one ledger row, and
that row is `VOID`.** Attempt 3 would cost ~17.6 h — **59% of `W37`'s fresh
30 h** — and the builder's own replay of the committed successor gate says the
most likely verdict is `SPLIT-PENDING`, i.e. *"still converging"*, i.e. a third
non-verdict. **This is not on your desk and I am not putting it there**: it is a
dispatch question, it is dated to the Review for 09-14, and the builder has
twice declined to pre-empt it. You are seeing it because ~51 GPU-hours for zero
verdicts on the project's largest single unblock (frees 35 specs, blocks 38) is
a number you should not first learn about afterwards.

**4. NO-DECISION — what I could not find a guard for.** The dispatch path that
spends that GPU (`scripts/dispatch.sh`, 61 lines) refuses exactly two things:
an unpushed HEAD, and a held lock. **No budget check, no precondition check, no
authorisation check.** Every constraint currently keeping `D1.0` unspent lives
in prose that no code reads. It has held — the builder refused twice in eight
hours, voluntarily, and wrote down why both times. I have routed the repair as
**B1** and it needs nothing from you. I am reporting it because *"it held
because the organ was conscientious"* is a different safety property from *"it
held"*, and you are entitled to know which one you have.

**5. NO-DECISION — the sentence I most want you to read, and it is the same one
as yesterday, which is the point.** `coverage` still exits `rc=2` on **four
claim-dead constitutional commitments** — *smell* (which you named
constitutional), *balance*, *"he builds a shelter"* and *"too cold kills him."*
`champions` now reports **four architectural seats nobody can ever win**,
including *language grounding*, which joined yesterday when `LG.03` was honestly
foreclosed. The union is **seven** distinct commitments or seats with no live
path, and until B3 ships, no single number in this repo prints it. Nothing was
hidden and nothing should be deleted — every one is a park or foreclosure that
was evidence-backed and correctly recorded. **The repair for all seven is the
same: successor specs.** That work is routed (DUE 09-16, tenth day) behind a
review desk whose own drain reads **UNBOUNDED**, carrying 14 rows due today
against a measured capacity of 6.

Six days at 108/245 is not a builder problem. The builder had its best night in
a week. It is that the ladder has run out of rungs that can be climbed without a
redesign, and the redesigns are queued behind a desk that cannot say when.
