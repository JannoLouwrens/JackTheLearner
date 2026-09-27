# OVERSIGHT.md — the overseer's current-state report

> Written by the overseer organ. **Current state, not a log** — each audit
> rewrites this file. Findings are ranked by how much damage they do to the
> trustworthiness of the ledger, not by how interesting they are.

**2026-09-27 12:37–12:5x UTC — the 125th audit.** Six hours after the 124th
(06:39–06:52). The window is the builder's six slots `07:07`–`12:07`, which
produced **eighteen commits** and left demonstrated **flat at 107/254** (42.1%).

Instrument exit codes, re-derived this sitting and not inherited:
`coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run status` **2**, `run ratchets` **2**, `run review-queue` **0**,
`run verify` **0 — and RANK 1 is that this exit code is a constant**.

Ratchet delta vs HEAD's committed readings, quoted as a DELTA: **6 MOVED**
(`fail_unowned_owned_forms` queue-row 29 → 30, `review_queue_net_arrivals`
26 → 31, `review_queue_piled_on` 3 → 4, `review_queue_violation_forms`
`{OVERDUE:1}` → `{}`, `review_queue_violations` 1 → 0, `unreachable` 95 → 96);
1 day-rolled (`cpu_foreclosed_now`). Floors: **3 ABOVE** — all three inherited
from the 124th, all three re-derived here and none newly caused in my window.

---

## VERDICT: INTEGRITY RISK — narrowly and precisely: the record itself is sound (106 of 107 PASSes re-judged from the record with **0** disagreements, 0 blind gates, every `commit` resolving in git), but the channel that *guarantees* that going forward has no live enforcement left. `run verify`'s exit code is the constant `return 0`; it hardcodes `exclude=("T0.18",)`, and at 12:18 today T0.18 became the one entry that fails it; and its gating spec `T0.18` cannot be re-run because `T0.13` is FAIL. Three silencers on one channel, two of them 48 days old and harmless until this morning. No dishonesty is alleged — the builder disclosed the class change in its own commit and queue row within the same slot; what reaches no page is that nothing can now report it.

---

## RANK 1 — the ladder's re-derivability guarantee is unenforced in all three of its channels at once, and it went that way at 12:18 today (HIGH, and it is the trust anchor)

`T0.18` is *"Every PASS is re-derivable from the record, and every control is
read."* It is the spec my own charter §1 leans on. Today it became the one
standing PASS in the ladder whose gate does not replay — and each of the three
things that could have said so is switched off.

**Leg 1 — the entry does not replay.** `eba3e58`/`3eddd91` added
`refused_refusal` to `T0.18`'s `_check` (marked *"armed 2026-09-27,
strengthen-only"*, and it is a genuine strengthening). Its standing row is
attempt 6, `ran_at 2026-08-30T08:18:55`, commit `7ffd961`. I enumerated that
row's metric keys against the keys `_check` reads:

```
keys _check reads that are ABSENT from the row:  ['refused_refusal']
```

so replaying the gate raises `KeyError`, not `False`.

**Leg 2 — the instrument named for that class excludes exactly that entry.**
`run.py:3455` is `scan(collect(ledger, exclude=("T0.18",)))`. I ran the same
scan both ways, read-only, no ledger write:

```
                    with the exclusion      without it
unevaluable_gates            0                  1
unevaluable_detail          ''          'T0.18(KeyError)'
self_excluded_entries        1                  0
verdicts_rejudged          106                106     (0 disagreements both ways)
```

The exclusion has been in the file since `2cd0289`, **2026-08-10** — T0.18's
first PASS. It is legitimate *inside T0.18's own run* (the entry is written
after the scan) and the CLI inherited it. It was harmless for 48 days. At 12:18
today the excluded entry became the only failing one, and an inherited argument
turned into a silencer.

**Leg 3 — the printed line is not the exit code, and the exit code is a
constant.** `cmd_verify` (`run.py:3445–3497`) contains exactly one `return`
statement and it is `return 0`. There is no branch on any finding: 0
disagreements and 100 disagreements both exit 0. That is not the module's
idiom — `run.py` returns 1 or 2 from a dozen other commands. Enforcement is
delegated to the spec (`"gated as T0.18"`), which brings us to:

**Leg 4 — the gating spec cannot run.** `T0.18.depends_on = ['T0.08','T0.13']`
and `Ledger().unsatisfied(T0.18)` returns `[('T0.13','FAIL')]`. The runner
refuses it. `run status` already prints this under STEERING-PAGE ORDERS:
`SUBJECT T0.18  BLOCKED — T0.13 (FAIL)`.

**And the same certificate is independently indicted by three other
instruments, none of which talks to the others:**

- `run stale`: `T0.18 PASS` — `impl_sha b18c290e1beee2e6 → 42c97c06b47f2328`,
  *"what moved: its own test file; MOVED: experiments/verify.py; IMPL_DEPS
  ITSELF changed [] -> ['experiments/verify.py']"*.
- `coverage`: `T0.18 ← T0.13 FAIL` is one of the 5 `PASS-ON-DEAD-DEPENDENCY`
  pairs, above the floor of 3.
- `T0.13`'s own `_check` gates `unevaluable_gates == 0` and `stale_gates == 0`,
  so it *does* see this — and `T0.13` is FAIL, so the signal is buried inside a
  red row whose repair is owed by the desk on 10-05.

**What I am NOT claiming.** Not that the builder hid it: `unevaluable_gates
0 → 1` is in `3eddd91`'s commit body and in a BUILDER-TRACE on
`t013-latently-red-28-disarmed-keys`, and the reclassification argument
(*"'this scan did not audit that gate' is true of a row predating the gate"*)
is correct. Not that the 106 re-judged verdicts are in doubt — they are clean.
Not that the detector is broken: `assert_detector_works()` plants `FIX.tuple`
and asserts `unevaluable_detail == "FIX.tuple(CheckReturnInvalid)"`, so the
detector demonstrably sees the class. **The defect is that the live instance
cannot reach any reader**, and that the 124th audit and I both quoted
*"`run verify` EXIT 0"* as evidence of cleanliness when that exit code carries
no information at all.

Repair is in FOR THE BUILDER 1–2 and is cheap: both halves are argument/scope
corrections to an existing checker, not a new organ.

---

## RANK 2 — today's probe-C repair moved two control-less PASSes out of a gated counter into one with no floor, and `D35` clause 2 forbids the builder from flooring it (MEDIUM-HIGH)

The act is right and was pre-priced by the desk; the consequence is mine to
report.

Before 12:18, `T0.01` and `T0.10` counted in `declared_control_never_ran`,
which `T0.18`'s `_check` gates at `== 0`. After `eba3e58` they are
`NoControlByDecision` — falsy by type — so that counter reads **0** and the
only class that still names them is `no_control_specs` = **2**. I verified its
standing:

- `verify.py` declares exactly one constant, `UNDECLARED_CONTROL_BUDGET = 0`.
  There is no `BASELINE_*` for `no_control_specs`.
- `FLOORED_CLASS_TOOLS = ("coverage","champions","review_queue","decisions")` —
  `verify` is not in it, so the floored-class scan cannot see it either.
- `T0.18._check` does not read `no_control_specs` in any conjunct.
- `run verify` prints it under a `?` marker and returns 0 (RANK 1, leg 3).

So the population that used to sit under a gate now sits under nothing. My own
charter §1 is *"a PASS whose control was never run is a claim without
evidence"*; my independent sweep of all 107 standing PASSes puts that class at
**2 — the first time it has been non-zero on the declaration question** (0 with
a missing `commit`, 0 whose commit is absent from git, 2 not declaring a
control: `T0.01`, `T0.10`, both with empty `control_metrics`).

**Both defences the builder offered are real and I checked them.** A hand-typed
imitation does not claim the exemption — I measured
`declares_a_control("NONE, BY DECISION (typed): trust me")` → **True**, a
promise. And the exemption costs a visible `registry.py` diff. **What neither
defence gives is a number that moves.** The repair — one `take()` line joining
`no_control_specs` to `ratchet_live` with a floor of 2 — is a **new ratchet**,
which `D35` clause 2 forbids by name: *"no new audit organ, checker or ratchet
may be built … nothing joins them"*. So it is the owner's, not the builder's,
and it is appended as evidence to `D35` this sitting.

---

## RANK 3 — the guard widened this morning to catch the refusal-that-ran-a-control reports that case as `"is None"`, because its own message branch is unreachable (LOW, one word wide, and it is inside the commit that claimed every truthiness reader became correct)

`protocol.py:3925–3931`, the `run_spec` pre-compute guard widened today
(strengthen-only, and the widening itself is correct):

```python
if control_fn is not None and not declares_a_control(spec.control):
    raise UndeclaredControl(... + ("declares NO control, by decision"
                                   if spec.control else "is None") + ...)
```

`declares_a_control` is False in exactly two cases, and in both of them
`spec.control` is falsy — `None`/`""`, or a `NoControlByDecision` whose
`__bool__` is False. Measured:

```
bool(refusal) = False          message branch picked: "is None"
```

So the `"declares NO control, by decision"` branch is dead code, and every
refusal-with-a-control — the precise case this guard was widened to catch —
tells the reader the field `is None` when it in fact holds a declared refusal
with a recorded reason. The two have different repairs. `T0.18`'s control
asserts the guard *fires* (`refused_refusal == 1.0`); nothing asserts it names
which defect it caught. One-word fix in FOR THE BUILDER 3.

---

## Sections 1–7

**§1 LEDGER INTEGRITY — clean, re-derived independently of `verify`.** All
**107** standing PASS rows: **0** with no `commit`, **0** whose commit is absent
from git (resolved one `git cat-file -e <sha>^{commit}` per row), **0** with no
registry spec. `run verify`: 106 re-judged, **0** verdicts that no longer
re-derive, **0** gates that ignore their control, **0** declared-but-unrun
controls, **0** unavailable entries. The two exceptions are the declared
refusals in RANK 2 and the one unreplayable entry in RANK 1.

**§2 THRESHOLDS AND CONTROLS — no loosening, and one case worth naming because
it points the other way.** Every numeric change in the window is a
strengthening or a disclosure:

- `protocol.py` / `verify.py` / `T0.18` (RANK 1–3): predicate moved, `_check`
  gained two conjuncts, `run_spec` gained a refusal. `UNDECLARED_CONTROL_BUDGET`
  still 0; `declared_control_never_ran` still gated `== 0`; `spec_sha` measured
  unmoved for both re-declared specs. No bar moved in either direction.
- **`XL.01`'s estimator asymmetry (`b3233a5`) is a STRENGTHENING held on the
  claim, not a loosening, and I checked the direction rather than the word.**
  `search_time_ratio` is gated `<= RATIO_MAX 0.5` — lower is better. The claim's
  mean-of-per-seed form reads **1.0034**; the equal-N pooled form the control
  uses reads **0.7286** on the identical recorded numbers. The claim therefore
  carries the *harsher* statistic, and the available "repair" (pooling it) is
  the loosening move — which the builder's own routed row says out loud and
  refuses to take by side effect (`xl01-claim-ratio-kept-the-per-seed-form-the-
  control-was-pooled-off`, DUE 10-07, `WAITS-ON: aggregate-hides-worst-seed`).
  Attempt 3 FAILs under either estimator. **This is the correct handling.**
- `274d987` inverted the builder's own routed row against itself: the salt
  detector's first live hit was its own cold cache (`c_fixture_ok` 0.0 at
  `PYTHONHASHSEED=0` too, 40 of 40 experiment metrics bit-exact), so
  `XL.01`'s rig is innocent and a bill priced against it is not owed.

**No control deleted or weakened, no `_check` gained an `or`, no seed count
reduced, no assertion removed, no `_GATES_FROZEN` flipped.**

**§3 DRIFT — nothing serves no GOAL.md sentence; the converse is the finding.**
Window work traced: `XL.01` attempt 3 + estimator diagnosis → *"he lives, he
dies, he remembers … life N+1 must be measurably better than life N"*;
`PS.05`'s zero-parameter inversion (r² 0.9994 worst-seed ceiling) → *"hot,
heavy, far, tiring, worth-it"*; `T0.18`/probe C and the `IMPL_DEPS` sweep →
*"really learning, not appearing to learn"*. **2 of 5 units were Jack's
science and neither moved a claim** — both were ceiling/estimator diagnoses
that make a FAIL more honest. Demonstrated 107 → 107 across six slots.
The converse, at HEAD: **0 of 32 commitments uncovered** (at floor since
09-19), but **3 CLAIM-DEAD** (smell, shelter/building, thermal-kills) and **14
more with live claim specs and nothing passing** — 17 of 32 of the owner's own
commitments have zero passing claims. Curiosity 2 of 12; one brain/unison 1 of
28. `coverage` also still prints **4 NEW unrunnable GOAL.md citations**
(`GEN.02/03/06/09`, all `welded<-LC.07`) — already found and dated by the 114th
audit (`gen-four-reparented-to-a-decision-that-had-already-closed`, DUE 10-01),
reported here as **standing and unrepaired**, not rediscovered.

**§4 BUILDER LIVENESS — alive, disciplined, and the board is empty.** **24
iteration ends in 24 h, 23 `rc=0`**, one `rc=124` (the 02:07 slot's inherited
timeout, handled correctly by the 03:07 slot). Demonstrated **111 → 107** over
24 h; all four losses are standing PASSes the builder re-bought to honest reds
unprompted. **Dark-slot streak 0.** `coverage` QUEUE DEPTH: **6 dispatchable,
all 6 VOID, 0 FRESH** — the empty board for the 30th consecutive slot, every
cost class `NOT FILLABLE`, three of them with *"nothing to implement, nothing
to pilot"*. The builder manufactured nothing and disclosed the refusal each
slot. **The builder is not the constraint.**

**§5 COMPUTE HONESTY — GPU clean, and the 124th's CPU finding RECURRED within
six hours with a bigger number.** GPU: `2026-W39` opened today; `gpu_budget.json`
carries **no W39 job at all** — 0.00 h charged of 30 free, expiring Saturday
2026-10-03. `W38` closed at 0.918 h of 30. Third consecutive week at risk;
every GPU class is `NOT FILLABLE`, so no dispatch should be manufactured.
`gpu_hours_no_verdict` unchanged at **48.42 h**, of which `D1.0` is 33.78 h
across 2 attempts and **0 verdicts**. CPU: today's metered day is
**3226.65 s** (`LT.02` 1610.40, `XL.01` 1168.90, + 8 Tier-0 rows), 0 overruns.
**Against that, commit `274d987` discloses `~600 s CPU` for a hand-invoked
three-leg measurement, and `cpu_budget.json` has no entry for it** — the
124th's RANK 2 measured 482 s outside the meter yesterday; the same exemption
(*"a module invoked BY HAND … a human at a shell is the owner's lane"*) carried
~600 s more today. Not misconduct, nothing near a ceiling (3226 s of 57600),
and the exemption is declared in the right place — it is the exemption's
*stated party* that is wrong, which is the owner's call and already on the desk.

**§6 STUCK DECISIONS — `decisions --check` EXIT 1. Nothing for me to escalate
and nothing for me to arm.** **0 `MEANS-ESCALATED`, 0 `UNDECLARED`, 0
`OVERDUE — DEFAULT IS DUE TO FIRE`, 0 `UNROUTED-OWNER-ASK`, 0
`VANISHED-OWNER-ASK`.** I looked for an `UNDECLARED` to arm as the mandate
requires; the class is genuinely empty. `D37` is armed and due 10-04.
**Three `CONDUCT-DESK` entries are stale** — `D33` by 4 days, `D35` by 3,
`D38` due 10-04 — all the Review's own conduct, and `D33` is the entry whose
authorship that desk **DECLINED** on today's `PROGRESS.md`, so the sole ratchet
red (`decisions_default_action_expired` = 1 vs floor 0) cannot be cleared by any
desk. I re-tested the 123rd/124th's reason for not repairing it and it still
holds: `DEFAULT-ACTION-EXPIRED` is computed from the text *inside* `D33`'s own
`DECIDE:` block, which this organ may not edit. The three
`PROGRESS.md` owner-asks are all correctly attributed to reaching desks
(`D10`, `D28`, `D33`) — and this is the first audit in a week where that reader
is reading a page published **today** rather than a stale one.

**§7 BAKEOFF HYGIENE — no finding.** `DECISIONS_RESOLVED.md` is unchanged in my
window. No VOID treated as a verdict there; the one place a VOID seats a
champion is declared and already indicted by `champions --check`
(`VERDICT-IS-A-VOID`, Learning core, `LC.03`), which is `D29`'s recorded debt.

**§ARCHITECTURE — `champions --check` EXIT 0, ratchet ok, every class at its
floor** (0 phantom arenas, 2/3 unfalsifiable, 4/4 unwinnable, 2/2 unverified
verdicts, 3/3 trigger debt, 1/1 kindless discharges). The two `UNCONTESTED`
seats I owe a schedule (Vision encoder, PLASTIC ONLY) both turn on `PL.02`,
whose repair is already dated (`pl02-void-gate-quantifies-over-its-own-nulls`,
DUE 10-05). Recording that rather than writing a second schedule.

**§QUEUE — `review-queue` EXIT 0, 0 violations, and the drain is the story.**
57 OPEN / 3 HELD / 81 live; oldest live 34 d; **arrived 39 (5.57/cycle) against
disposed 8 (1.14/cycle) — drain UNBOUNDED**, net arrivals 26 → **31**. The
OVERDUE class is empty because five rows were re-dated 11–15 days out this
morning; `disposed`, the live count and the drain are byte-identical across
that commit (the 124th's RANK 3). **13 live dated rows fall due by the
consumer's next cycle against a measured capacity of 6, and 7 of them cannot
be discharged by it.**

---

## FOR THE BUILDER

1. **`run verify` must stop inheriting `T0.18`'s in-run self-exclusion.**
   `run.py:3455` passes `exclude=("T0.18",)`. That argument is correct when
   `T0.18` runs its own scan (the entry is written afterwards) and wrong for the
   CLI, where the 08-30 row **exists** and is the standing certificate. Drop it
   at the CLI call site — or, if you prefer to keep one code path, print BOTH
   numbers (`unevaluable_gates` with and without) so the excluded entry cannot
   be the invisible one. Expect the honest reading to be
   `unevaluable_gates = 1`, `unevaluable_detail = T0.18(KeyError)`; that is the
   true state and it is the point. This is a scope correction to an existing
   checker, not a new organ, so `D35` clause 2 does not reach it.
2. **`cmd_verify`'s exit code is the constant `return 0`** — one `return` in
   the whole function, no branch on any finding, while a dozen other `cmd_*`
   return 1 or 2. Two audits in a row (the 124th and me) quoted *"`run verify`
   EXIT 0"* as evidence of a clean record; it means only that the tool ran.
   Return non-zero when `verdict_disagreements`, `control_blind_specs`,
   `declared_control_never_ran`, `unevaluable_gates` or `unavailable_entries`
   is non-zero, with `undeclared_control_ran` against its budget as today. If
   you judge that enforcement belongs solely to `T0.18` and the CLI should stay
   reporting-only, then say so **in the function's docstring** and note there
   that `T0.18` is `BLOCKED — T0.13 (FAIL)`, so that a reader cannot mistake a
   constant for a verdict. Do not change any threshold to make it green.
3. **One word in `protocol.py:3929`.** The widened guard's message branch tests
   `spec.control`, which is falsy for every case that reaches it, so the
   `"declares NO control, by decision"` arm is dead and a refusal-with-a-control
   is reported as `"is None"`. Test `is_control_refusal(spec.control)` instead.
   While you are there, the same commit's claim that *"every `bool(spec.control)`
   in the repo became correct with no edit"* is true everywhere except this one
   message, which that same commit wrote — worth a line in the docstring so the
   claim is not inherited as fully verified.
4. **Do NOT re-run `T0.18` to clear RANK 1, and do not re-run anything to clear
   RANK 2.** `T0.18` is `BLOCKED` behind `T0.13`'s FAIL; its re-buy is owed
   *after* the per-key adjudication on `t013-latently-red-28-disarmed-keys`
   (DUE 10-05), by the desk, not by you. Items 1–3 make the state visible; they
   do not repair it, and pretending otherwise is the failure mode.

## FOR THE OWNER

**1. `D35` clause 2 now forbids the only repair to a hole `D35` did not
anticipate — one line, and it is the opposite of organ-building.** Today's
probe-C repair (correct, and pre-priced by the desk's own queue row) moved the
two control-less PASSes `T0.01`/`T0.10` out of `declared_control_never_ran`
(gated `== 0` by `T0.18`) into `no_control_specs`, which has **no floor, no
committed reading, no gate and no exit code** — I verified all four. Growth in
*"PASSes resting on a gate never shown able to report the bad case"* is now
invisible to every instrument this project owns. The repair is one `take()`
line plus a floor of 2, in the exact idiom five `decisions.py` classes joined
on 09-26 — and clause 2 says *"no new audit organ, checker or ratchet may be
built … nothing joins them"*, so the builder may not write it. This is the
third realised cost of clause 2 (the 116th audit recorded two) and it is the
first where the forbidden act is a FLOOR rather than a checker. An EVIDENCE
ADDENDUM is appended to `D35` with the measurements. The 116th audit's option
**(b)** — *exempt truthfulness repairs and floors on EXISTING checkers from
rule 2* — would discharge this and needs no new decision entry.

**2. The creature-gate quota becomes unsatisfiable at the 13:07 slot — within
the hour — and this is the second time, not the first.** `D35` rule 3 allows
`"none"` at most twice running. This slot was NONE #2 of 2. Of
`T2.01`/`XL.01`/`T6.01`: `T2.01` is settled FAIL with both repair lanes
desk-owned and prohibited to the builder by name; `T6.01` is unimplemented
behind `T4.05 ← T4.04 ← T2.01`, itself behind `T1.08` (FAIL); and `XL.01` is a
**settled FAIL run this morning**, so rule 3 resolves to *re-run a settled
FAIL*, which `run next` forbids. The builder wrote this into its journal and
commit rather than discharging it by re-rolling `XL.01`, which is the correct
conduct. The last time this happened (09-24) it produced **20 consecutive
recorded violations** of a rule filed one day earlier. The 116th audit already
put three one-line repairs on your desk — (a) a reachable release condition,
(b) the clause-2 exemption above, (c) confirm the freeze is meant to be
unbounded and mark the accrued violations absorbed. **None has been ruled, and
the accrual restarts at 13:07.** This is the only perishable item on this page.

**3. NO-DECISION, standing report: 30.0 free Kaggle GPU-hours opened today,
0.00 charged, expiring Saturday 2026-10-03 — the third consecutive week at
risk, and no dispatch should be manufactured for them.** Every GPU cost class
reads `NOT FILLABLE — the repair is a REDESIGN`, and both live routes run
through `T1.08` (FAIL, blocks 45), whose repair is undesigned until 10-02.
`D1.0` still holds **33.78 GPU-hours across 2 attempts and 0 verdicts** — the
largest unredeemed spend in the project, unmoved since 09-18.

**4. NO-DECISION: the CPU day meter's exemption recurred within six hours of
being reported.** The 124th audit measured 482 s of the loop's own CPU outside
`cpu_budget.json` under the *"a human at a shell"* exemption; commit `274d987`
discloses **~600 s more** today, in a day metered at 3226.65 s. Nothing is near
a ceiling and nobody broke a rule — the exemption names a party that is no
longer the party using it, which is a scope decision and forbidden to the
builder by clause 2. Reported as a **confirmed recurrence with a larger
number**, not as a new finding.

---

## §8 THE HONEST ANSWER: no — and today the reason is unusually specific

Six slots, eighteen commits, demonstrated flat at 107, and the two units that
were about Jack were both diagnoses that made an existing FAIL more honest
rather than teaching him anything. Nothing in this window taught him to see, to
want, or to remember.

What the window *did* buy is real and I do not want it flattened: a detector
that had been counting a refusal as a promise for 48 days was found and
repaired, the repair was found to be the option a queue row had explicitly
REFUSED and was converted **in the same slot before any certificate was bought
against it**, a salt-lottery scare was inverted against the row that raised it,
and an estimator asymmetry in a live claim was measured, quantified at 55% of
the distance to its bar, and routed rather than exploited. That is a loop
auditing itself honestly at a standard most projects never reach.

And it is also the RANK 1 finding in miniature. The thing this project is
building right now is the apparatus, and the apparatus has become intricate
enough that its own trust anchor could lose all three of its enforcement
channels in one morning without any instrument saying so — while the jungle
Jack is supposed to live in still does not exist, its design has now lost five
consecutive Sunday sittings and been formally DECLINED by the desk that owed
it, and 17 of the owner's 32 constitutional commitments have zero passing
claims. We are not closer to a curious humanoid than we were yesterday. We are
closer to being able to prove it when we finally are — and that is worth
something, but it is not the goal, and the gap has now been legible on this
page and on `PROGRESS.md` for two consecutive sittings.
