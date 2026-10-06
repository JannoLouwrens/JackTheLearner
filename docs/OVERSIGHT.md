# OVERSIGHT.md — the overseer's current-state report

> Current state, not a log. Each audit rewrites this file.

**2026-10-06 06:37 UTC — 143rd audit.**

## VERDICT: DRIFTING

The ledger's integrity checks are clean and section 2 is clean, which I say
plainly because it is true and it is the valuable half. The drift is elsewhere
and it is the same drift the Review named on 10-04, now two days older: **nine
days without an edit to the thing that is supposed to learn.** The new finding
is worse than drift — a field two separate meters read as `MEASURED` is wrong
by four orders of magnitude on the newest family of specs, and one of those
meters is the guard against a hash-salt lottery that was built because such a
lottery came one count-tie from deciding a seat.

**Instruments, every one re-run immediately BEFORE this file was committed and
not quoted from the top of the sitting** — the 06:37 Review collision makes a
stale reading the default failure here, and RANK 3 below is about exactly that
mistake: `coverage` **2**, `decisions --check` **1**, `champions --check` **0**,
`run status` **2**, `run review-queue` **2**. *(My own first pass misread
`coverage` as 0 by taking `head`'s exit status through a pipe rather than the
tool's; corrected here before commit. `coverage` exits 2 on the three
above-floor ratchets — `decisions_default_action_expired` 1,
`pass_on_dead_dependency` 6, `unreachable` 96 — none of them new.)*

**Concurrency disclosed:** the Review DAILY committed five acts (`21d0994`,
`7d54204`, `3d04272`) while this audit was running and `docs/REVIEW_QUEUE.md`
was dirty in its hands at commit time. I committed only `docs/OVERSIGHT.md` and
`docs/LESSONS.md`, by name. Every queue number on this page was read before
those acts landed and the Review's own page is the later authority on them.
`decisions --check` still exits 1 after them, so RANK 2 was not claimed by that
sitting.

---

## RANK 1 — `duration_s` is a fiction for the pool-prefetch class, and TWO meters consume it as a measurement

This is the most damaging finding of the sitting, and neither organ has seen
the half that matters.

**The number.** `W1.01`'s PASS row, recorded 05:59:23 today, carries
`duration_s` **0.22**. The same run billed **2,479.79 s** to
`experiments/cpu_budget.json` under its own label, and the row's own metrics
record `wall_s` **1,154.89**. The declared duration is **11,272× under** the
cost.

**The cause, at source.** `experiments/tests/w1_01_passivity_dies.py:392`
`run()` executes all 12 arm tasks in a 3-worker `Pool` and memoises them into
`_CACHE` *before* calling `run_spec` at line 407. `run_spec` therefore times a
dictionary read. This is the deliberately-adopted "W1.02 pool pattern" and it is
a class, not an instance: `w1_00:481`, `w1_01:403`, `w1_02:369`, `w1_04:383`.
**W1.04's recorded run is in flight as I write (pid 514742, 3 workers verified
at 98.8% CPU, 17 min elapsed) and will be the fourth instance.**

**Consumer A — the tenant-protection day gate, which nobody has named.**

    child_estimate_s(W1.01) -> (10.88, 'MEASURED 0.22s x4 + 10s')
    child_estimate_s(W1.02) -> (10.68, 'MEASURED 0.17s x4 + 10s')
    child_estimate_s(SO.07) -> (36816.04, 'MEASURED 9201.51s x4 + 10s')   <- honest
    child_estimate_s(T1.01) -> (8340.24,  'MEASURED 2082.56s x4 + 10s')   <- honest

`gate_cpu_child` would admit a 2,480 s child as an 11 s one against the
57,600 s day ceiling — a **228× under-estimate carrying the provenance word
`MEASURED`**, on a box with paying tenants. `cpu_budget.py:250`'s docstring
promises *"Clamped at the enum by construction, so the projection may only
TIGHTEN an estimate and this gate can never refuse something it would have
admitted before."* That clamp is **one-sided**: it bounds over-estimation and
silently assumes `duration_s` honestly lower-bounds the spec's cost. For this
pattern it does not, and the enum lane (`W1.04` today: 54,000 s) is the
*accurate* one — the MEASURED lane is strictly worse than having no measurement.

Measured across the ledger, rows whose `duration_s` is under a tenth of a
measured real cost above 60 s: **W1.01 11,272×, W1.00 4,216×, W1.02 3,741×.**
(`LG.13`/`LG.14`/`T0.2x` also appear on that scan and are **not** instances —
LG.14's 1.3 s recording run is an honest replay against a separately-billed
verdict cache, and the T0.2x ratios are my aggregation summing 20 re-buys in a
day. The W family is the real class.)

**Consumer B — the hash-salt differential, which the builder saw the symptom of
and misdiagnosed.** `protocol.py:3972`:

    tmo = max(120.0, 3.0 * float(elapsed_s))

with the comment *"the ceiling is 3x the run it is differentiating."* With
`elapsed_s` = 0.22 that is **120 s**. The salt child re-runs `_experiment`, not
`run()`, so it starts with a **cold `_CACHE`** (`w1_01:284`) and does all 12
rollouts serially in one process — roughly 3× the 3-worker wall, so ≳3,400 s
against a 120 s ceiling. That is the ledger's **first-ever hash-salt
differential `TimeoutExpired`** (11 deciding metrics, reporting-only), now on
W1.01's row.

**Why this is worse than a metering slip.** `protocol.py:3718` records why the
instrument exists: LG.10/LG.12 recorded `swap_agree` as a function of
`PYTHONHASHSEED` — *"an unrecorded, unreconstructible per-process salt"* — and
*"the same statistic was one count-tie away from deciding a SEAT."* Its coverage
is now **anti-correlated with run cost**: absent on exactly the certificates
that are most expensive to re-derive by hand, and guaranteed-absent (not
probabilistically absent) on every spec using the pool pattern. It is
reporting-only and lands in a prose `message` string, and **there is no counter,
ratchet or exit code for salt-differential outcomes anywhere** — grep of
`coverage.py` and `run.py` returns nothing. "Which certificates have no salt
reading" is not a number this project can print.

**Honest scope, stated rather than inflated.** Since the instrument went live
2026-09-26 there are 28 rows: **16 CLEAN, 2 non-deciding blemish, 1 ERRORED
(W1.01), 9 with no note at all.** One instance, not an epidemic.

**But the builder's deferral is wrong on its own terms.** The 06:07 journal says
the finding is *"routed as a row only if W1.04 reproduces it."* The timeout is a
deterministic function of `duration_s` and the cold-cache replay, both readable
at source in under a minute; a second instance adds no information and costs
another expensive certificate its salt reading. And the builder did not see
Consumer A at all.

---

## RANK 2 — the project's ONLY broken ratchet has stood 13 days behind a reason that does not cover the repair the instrument itself names

`decisions --check` EXIT 1: **`RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED,
baseline 0`**, sole member `D33`, reading ≥1 since 2026-09-23.

The Review's 10-04 `FOR THE OWNER` item 3 explains the standing red to the
owner: *"the 10-02 Review established its default is MOOT rather than merely
expired — its object went terminal when `w1-world-edit-window` was stamped
`DECLINED`, so **no desk can clear this by firing anything.**"*

**That sentence is true and it answers the wrong question.**
`DEFAULT-ACTION-EXPIRED` is not about firing. `decisions.py:227`: *"the
default's prose names a date at or before its own firing date."* The tool prints
**two** repairs and the second fires nothing (`decisions.py:1402`): *"declare
whose it is with `(CLOCK: <whose>)` beside that date — a provenance date that
says so stops reading as a command."*

**The repair is precedented twice inside this very register.**
`decisions.py:666` records that this class shipped at 1 (`D22`) and *"shrank to
0 on 2026-09-05 (72nd audit B2) ... when `D22` DECLARED both of its dates
provenance with `(CLOCK: D15)` and `(CLOCK: consequence)`. **Shrink by
declaration, exactly as prescribed: the entry says whose clocks those are, and
the instrument was not tuned.**"* And `DECISIONS_NEEDED.md:8840` already carries
`(CLOCK: w1-world-edit-window ...)` on another default.

**And D33's date qualifies cleanly.** Its default is *"(i) RE-DATE ONCE MORE, TO
2026-09-23"*, and D33's own NOTE says that act is **already done**: *"I re-dated
`w1-world-edit-window` to 2026-09-23 with a written cause and a stop-rule in
`c9aca70`, before this entry was written."* A date describing a completed act is
provenance, not a command — precisely the case the `CLOCK:` idiom was built for.

So the owner has been told their register's one broken ratchet is unclearable,
when the instrument, the idiom and two in-file precedents say it is clearable by
annotation, without firing, deciding, re-dating or moving anything. The
underlying W1 authority question stays entirely the owner's and is untouched by
this; what is wrong is the *reason given for inaction*, which is the thing this
organ exists to re-derive.

---

## RANK 3 — the Review quoted a ratchet its own act had invalidated six minutes earlier, in the paragraph that promises it did not

`pass_on_dead_dependency` reads **6** live against a committed reading of **5**
(recorded 2026-09-26) and a declared floor of **3**.

Recomputed by me at each revision rather than taken from either organ:

| revision | time | reading |
|---|---|---|
| `04f99d1` act 12 (the T1.11 strengthening) | 06:59:55 | 5 |
| `c7b4bb9` act 12b (**the T1.11 demotion**) | 07:00:30 | → 6 |
| `8a42c10` act 15 (**the page**) | 07:06:40 | 6 |

The new pair is **`T1.12 <- T1.11 (FAIL)`**. `docs/PROGRESS.md` act 15 says:
*"INSTRUMENT EXIT CODES, **every one re-run after the last act of this sitting
and not quoted from the top of it** (the 06:37 overseer collision makes a stale
reading the default failure here)"* — and then lists
`pass_on_dead_dependency` **5**. It was 6 when that line was committed, by the
sitting's own act, six minutes earlier. The builder caught the 6 at 23:07 on
10-05, roughly 16 hours later.

**The substance matters more than the bookkeeping.** `T1.12` "Flow matching
actually denoises" is a **standing PASS** (attempt 3, 2026-09-26T23:47:43) whose
**only** dependency is now FAIL, and **no queue row owns it**.
`t111-certified-loss-has-no-shipped-caller` (OPEN, DUE 10-16) owns T1.11's own
FAIL, not T1.12's certificate.

There is a real mitigating argument and it is available: T1.11's two original
conjuncts are byte-unmoved and still pass, only the new shipped-caller conjunct
failed, and T1.12's claim rests on the parity half. **But nobody has written
that down, and an unwritten mitigation is not a disposition.** No
`ratchets record` has blessed the move, which is correct — the floor may only
move in the commit that grew the number, with the reason in its growth log.

---

## The clean results, said plainly

**Section 1 — ledger integrity: NO FINDING, and I checked all of it.** 107 PASS
rows. **0** commits missing from git. **0** specs absent from the registry.
Every PASS whose spec declares a control has recorded `control_metrics` — **0**
exceptions. Only `T0.01` (repo imports clean) and `T0.10` (Kaggle round-trip)
declare no control, and both are harness plumbing where a control is meaningless.

**Section 2 — thresholds and controls over 7 days: NO FINDING, and I looked
hard.** 18 commits touched `registry.py`, `registry_expansion.py` or
`experiments/tests/`. The only numeric moves in the window: `CLF_EPOCHS`
300 → **900** (`5eba43e`, a *strengthening* — more training, justified by
attempt 1's VOID) and `_SEC_PER_SEED` (a metering constant, not a gate). Every
`or` added is a new VOID/FAIL lane — LG.14's four liveness/variety/mismatch
conjuncts — never an alternative route to PASS. **No `seeds=` reduced**; every
new spec is `seeds=3`. Two conjuncts were **added**: T1.11's shipped-caller
(`04f99d1`) and XL.01's pooled ratio (`e9086ac`), both pre-registered before the
run with still-FAIL declared in advance. W1.01's bar is read live from W1.02's
recorded PASS row (`quantum 0.0083681` appears in W1.01's own metrics), not
hard-coded at this desk.

**W1.01's PASS examined adversarially, because a spec registered with a
PREDICTED FAIL came back PASS.** It holds up. `gap_ok = 1.0, std 0.0` across
seeds [0,1,2] — a genuine 3/3, not a mean rescuing a bad seed. The benign-twin
control is quiet the required way (`gap -0.532` against the `0.0251` bar) with
`oracle_fed = 1.0`, so the mechanism demonstrably ran with the nutrition channel
dead. Both the builder's journal and the row keep the claim narrow — headroom
exists *on the oracle-vs-statue axis*, which is a statement about the WORLD, not
about Jack learning anything. That hedging is correct and I am recording it so a
later reader cannot inflate it.

**Section 4 — the builder: ALIVE, and the blackout is over.** 24 iterations in
the 24 h to 06:23 today, **23 ended rc=0**, one rc=124 (04:07) whose
complete-but-uncommitted W1.01 work the 05:07 slot inherited, semantically
diffed first and *verified rather than assumed*. `demonstrated` **106 → 107**.
`dark_slots` reads **0**, at floor.

> **The Review's 10-04 page is two days stale on its own headline and a reader
> will act on it:** it reports *"Builder dark 76 slots / 75.8 h, longest on
> record"* and *"first legal slot ≈ 2026-10-06 16:07 UTC"* in `FOR THE OWNER`
> item 2. The builder has in fact run every hour since 21:49 on 10-05. The page
> also still carries its `rc=124` INCOMPLETE seal from the 10-05 run.

**Section 5 — compute: the three-week drought broke.** `2026-W40` has **0.2708**
kaggle-h drawn of 30; **29.73 h remain and expire Saturday 2026-10-10.** That
0.27 h bought **T1.08 Step 1** and produced `eval_cv_pct` 0.52 — the first
*designed* GPU buyer to actually land in three weeks, after W37/W38/W39 expired
~83 h unbought. `gpu_hours_no_verdict` moved 49.49 → **49.77 h**; the +0.28 is
that job (T1.08 1 → 2 attempts, still 1 verdict; PROBE 4 → 5 jobs). Standing
waste unchanged and still the worst line here: **`D1.0` 33.78 h over 2 attempts
with ZERO verdicts**, and `UNATTRIBUTED` 6.32 h / 21 jobs (at floor 21).

> **PERISHABLE, and it is the one clock on this page that cannot be re-armed:**
> 29.73 free GPU-hours expire in 4 days, and their only designed buyer —
> `T1.08` Step 2b — is **the Review's to route.** The builder's 03:07 and 05:07
> slots each recorded that no routing had landed. Today's sitting is the last
> one with four days of margin.

**Section 3 — drift from the goal.** 63 commits in the last 24 h: 17
`experiments/`, 9 `docs/`, 5 `scripts/`, 1 `CHECKLIST.md`. **Zero touched
`UnifiedBrain.py`, `TrainingPipeline.py`, `playground.py` or `VirtualWorld.py`.**
The Review's 10-04 reading (249 commits / 7 days, Jack untouched) now extends to
**nine days**.

**But the honest counterweight, which I rank as the real change this week:** the
builder stopped only building apparatus and started pointing it at the creature.
T6.01's smoke measured that the shipped companion **cannot drive its own body** —
`apply_action` refuses a 17-wide action against 57 actuators on **177/177
frames** — and is **blind by scene wiring** (no `"eye"` camera). T4.04's smoke
measured that the two shipped objectives **share 136 gradient tensors**, so
interference is a real possibility rather than a decorative spec. W1.01 measured
that W0 has headroom on one axis. Those are measurements *of* Jack, not edits
*to* him — but they are the first numbers about the creature in a week, and
three of them are **bad news found rather than good news manufactured.** Every
one of the three traces to GOAL.md: a body with every sense (the blind scene),
one interconnected brain (the shared tensors), the world as teacher (W1.01).
Nothing in the last 24 h serves no GOAL.md sentence. I found no drift of the
"busywork" kind.

**GOAL.md commitments with no passing spec:** `commitments_uncovered` **0** at
floor; **`claim_dead` 3 for the tenth day — smell, shelter/building, thermal
("too cold kills him")** — routed as `D42` (armed, default (iv) HOLD,
`decide_by 2026-10-18`). 14 further commitments have live claim specs with
nothing passing, including touch, tool use, told-world, proprioception, sleep
and plasticity.

**Sections 6 and 7 — decisions and bakeoff hygiene: no new finding.** `D41` and
`D42` are armed, `decide_by 2026-10-18`, both monotone holds, and both correctly
declared as deliberately NOT the recommendation. All 4 `PROGRESS.md` owner-asks
route to real entries (D37/D33/D41/D42); `decisions_unrouted_owner_ask` and
`decisions_vanished_owner_ask` are both **0 at floor** — the `D15` scar is
holding. Firing transcription **31 of 31**. No `MEANS-ESCALATED`. No
`UNDECLARED`. No VOID treated as a verdict and no winner chosen inside a noise
margin in the window. Three `CONDUCT-DESK` entries are stale and flagged so they
cannot self-approve: **D33 (13 d), D35 (12 d), D38 (2 d)** — D38 is new this
window.

**Review queue.** 53 OPEN / 3 HELD / 30 DISPOSITIONED of 129 routed; oldest live
**43 d**; consumer ran yesterday. **`OVERDUE` 0 (was 5)** and
`review_queue_violations` **7 (was 14)** — the 10-04 sitting's disposal work was
real. All 7 remaining are `HOLD-ON-A-RESOLVED-BLOCKER` behind
`w1-world-edit-window` (DECLINED), deliberately **not** laundered, and I agree
with that refusal. But **`drain` is UNBOUNDED**: 86 live rows, arrivals exceed
disposals by 2 over the window, and `review_queue_piled_on` moved 4 → **9**.
Every dated promise in that file is downstream of a desk that is not keeping up.

**Champions.** EXIT **0**, every violation at its declared floor.
**`ARENA-MISSING` is 0** — the overseer prompt's *"8 seats today"* is a dated
example, not current state, and the next audit should not quote it. Standing bad
news unchanged and not re-litigated: the **Learning core** seat is held BY
VERDICT off a **VOID** (`LC.03`) with every pre-registered re-open trigger a
closed door, and the **World** seat is held BY VERDICT with neither a deciding
row nor a reachable rematch trigger.

---

## FOR THE BUILDER

1. **`duration_s` MUST STOP FEEDING TWO METERS AS IF IT WERE THE SPEC'S COST
(RANK 1).** Both consumers read the same wrong field and the honest number
already exists in the row.
   - **(a) `protocol.py:3972`.** `tmo = max(120.0, 3.0 * float(elapsed_s))` is
     120 s for every pool-prefetch spec, and the salt child replays
     `_experiment` against a cold `_CACHE`, so the timeout is **structurally
     guaranteed** — not a flake. Derive the ceiling from the run's real cost
     (`metrics['wall_s']` where present, or the spec's billed
     `cpu_budget` seconds), not from the wrapper's elapsed time. **Verify the
     cold-cache replay yourself at `w1_01_passivity_dies.py:284` before
     touching anything, and stop and route if it does not reproduce.**
   - **(b) `cpu_budget.child_estimate_s`.** The `MEASURED` lane returns
     **10.88 s** for a spec that costs ~2,480 s and is therefore strictly worse
     than the `ENUM` lane it is allowed to tighten. The clamp is one-sided;
     the invariant it protects ("may only TIGHTEN") assumes `duration_s`
     lower-bounds cost and that assumption is false for this class. The repair
     may **only tighten the gate** — it may not make `gate_cpu_child` admit
     anything it refuses today. Reporting-first is acceptable and probably
     right: print the provenance disagreement
     (`duration_s` vs billed seconds) wherever the estimate is read.
   - **(c) There is no counter for salt-differential outcomes.** `coverage.py`
     and `run.py` contain no reference. Add a **reporting-only, unfloored**
     reading in the `RATCHET COUNTERS` block — CLEAN / BLEMISH / DIVERGENCE /
     ERRORED / NO-NOTE over rows recorded since 2026-09-26 — so "which
     certificates have no salt reading" becomes a number instead of a grep of
     prose. **Do not gate anything on it** and do not let it fail a spec; the
     instrument is reporting-only by its own disposition.
   - **Route this as ONE queue row, now, not conditional on W1.04.** The 06:07
     journal's *"routed as a row only if W1.04 reproduces it"* defers on a
     question already settled at source.

2. **`T1.12`'s certificate needs a written disposition (RANK 3).** It is a
standing PASS whose only dependency went FAIL on 2026-10-04 and no row owns it.
The likely correct disposition is *"the parity conjuncts T1.12 relies on are
byte-unmoved and still pass; the new shipped-caller conjunct is orthogonal to
T1.12's claim"* — but **write it with the conjunct-by-conjunct reading
attached**, do not assert it. If that reading does not hold, the honest move is
to record T1.12's BLOCKED, and **say so rather than quietly preferring the
cheaper answer.** Do not `ratchets record` the 5 → 6 growth as a tidy-up; the
floor moves only in the commit that grew the number, with the reason in its
growth log.

3. **`D33`'s `(CLOCK:)` annotation (RANK 2) — ONE LINE, and it clears the
project's only broken ratchet.** D33's default names `2026-09-23`, equal to its
own `decide_by`, and D33's own NOTE records that the act was already performed
in `c9aca70`. Annotate that date `(CLOCK: c9aca70)` — or whatever provenance you
verify at source — exactly as `D22` did with `(CLOCK: D15)` /
`(CLOCK: consequence)` (`decisions.py:666`, and the live precedent at
`DECISIONS_NEEDED.md:8840`). This **fires nothing, decides nothing, re-dates
nothing and moves no threshold**; it declares whose clock a past date is.
`decisions_default_action_expired` should then read **0, at floor**, and
`decisions --check` should exit **0**. **This is D33's author's entry — the
Review may claim it first and should; take it only if the 06:37 sitting does
not.** If after reading D33 you judge the date IS a live command rather than
provenance, **do not annotate it** — say so and leave the ratchet red.

4. **The 10-04 `PROGRESS.md` page is stale on the builder's own status and
should not be inherited as fact.** It reports 76 dark slots and a first-legal
slot of 2026-10-06 16:07 UTC; you have run hourly since 21:49 on 10-05 and
`dark_slots` is 0. Do not edit the Review's page — just do not re-derive your
own board from it.

5. The 10-04 page's `FOR THE BUILDER` items 1–7 and the 135th–142nd audits'
orders are not re-ranked here; nothing above displaces item 1 of RANK 1, which
is the cheapest real finding on your board.

6. **THE CROSS-ORGAN COMMIT RACE SURVIVES ITS OWN DISPOSITION — see the
POSTSCRIPT, which is a finding this audit made by trying to commit itself.** At
06:51:47 the Review DAILY's `9a29031` absorbed this audit's staged
`docs/OVERSIGHT.md` and `docs/LESSONS.md` into its own changeset, so a
1,074-line rewrite of the overseer's page is attributed in git to *"Review DAILY
10-06 acts 6-7."* Content intact; attribution gone. **The existing defence is
one-directional:** "stage by name" protects the organ that commits, never the
organ whose staged index is standing when someone else runs a wildcard add.
`cross-organ-doc-race-voids-certificates` is already `ACTED` (`b4df9bb`), so
this is a surviving mode of a dispositioned class. **Reporting-first repair, and
do not make it a lock:** have each organ's commit helper stage an explicit
pathspec AND pass that same pathspec to `git commit -- <paths>`, so a commit can
only ever contain what its author named. **Verify the race at source before
changing anything** — read whichever helper `scripts/` uses for the Review's
commits and confirm it stages with a wildcard — and **stop and route if it does
not reproduce**, because an attribution fix built on a guessed cause is worse
than the race. The design of *who* owns a shared doc's write window is the
Review's, not yours.

---

## FOR THE OWNER

**1. Your register's one broken ratchet is clearable by annotation, and you were
told it was not.** On 10-04 the Review reported `decisions_default_action_expired
= 1` to you with the explanation that *"no desk can clear this by firing
anything"* (`D33`). That is true about **firing** and beside the point: the
class is about the default's prose naming a past date, and `experiments/
decisions.py:1402` names a second repair that fires nothing — declaring
`(CLOCK: <whose>)` beside the date. This register has already done exactly that
twice (`D22`; and a live instance at `DECISIONS_NEEDED.md:8840`), and
`decisions.py:666` records the precedent in the words *"shrink by declaration,
exactly as prescribed ... and the instrument was not tuned."* **No decision of
yours is being asked for and the W1 authority question in `D33` remains entirely
yours and untouched.** What this item reports is that a red light on your
register has stood 13 days behind a reason that does not cover the available
repair. Routed to the builder as item 3, with instructions to leave it red if
the date turns out to be a live command. **Nothing to rule on.**

**2. NO-DECISION — `D30`'s standing report. The builder blackout ENDED and the
GPU drought BROKE; both are better than the last page you read.** The builder
has run every hour since 21:49 on 2026-10-05 — 24 iterations, 23 `rc=0`,
`demonstrated` 106 → 107, `dark_slots` **0 at floor**. The Review's 10-04 page
still says *"dark 76 slots / 75.8 h, longest on record"*; that is two days stale.
**`2026-W40`: 0.27 of 30 free Kaggle GPU-hours drawn, 29.73 remaining, expiring
Saturday 2026-10-10** — and the 0.27 h bought `T1.08` Step 1, which returned
`eval_cv_pct` 0.52. That is the **first designed GPU buyer to actually land in
three weeks**, after ~83 h expired unbought across W37–W39. The perishable risk
is unchanged in shape: Step 2b is the Review's to route and had not been routed
as of 05:07 today.

**3. NO-DECISION — the thing I would want you to read if you read one line.** 63
commits in the last 24 h and **none** touched `UnifiedBrain.py`,
`TrainingPipeline.py`, `playground.py` or `VirtualWorld.py`; that is now nine
consecutive days. But the character of the work changed and it is worth your
knowing: this week the apparatus was finally **pointed at the creature**, and it
came back with bad news found rather than good news manufactured — the shipped
companion **cannot move its own body** (a 17-wide action refused against 57
actuators on 177 of 177 frames), it is **blind by scene wiring**, and the two
shipped training objectives **share 136 gradient tensors**. Those three facts are
worth more than the single green tick the week added, and all three are
downstream of the artefact question already on your desk as **`D41`**
(`decide_by 2026-10-18`).

**4. Nothing new is asked on `D41` or `D42`.** Both are armed, monotone, and
their defaults are correctly declared as deliberately *not* the
recommendation. `claim_dead` has read 3 for ten days (**smell**,
**shelter/building**, **thermal**) — `D42`'s subject, unchanged.

---

*Audit 143. Ranked by damage to the trustworthiness of the ledger. Sections 1
and 2 returned no findings and that is reported as a result, not as an absence
of effort: 107 PASS rows checked for a live commit, a registered spec and a
recorded control, and 18 commits of threshold diff read for a loosening that is
not there.*

---

## POSTSCRIPT — a finding this audit made BY TRYING TO COMMIT ITSELF

**At 06:51:47 the Review DAILY's commit `9a29031` ("acts 6-7") absorbed this
audit's two files.** I staged `docs/OVERSIGHT.md` and `docs/LESSONS.md` by name
at ~06:51; the Review's next commit landed with
`docs/OVERSIGHT.md +1074/-662`, `docs/LESSONS.md +56` and its own
`docs/REVIEW_QUEUE.md +101` in one changeset, and my `git commit` then reported
*"no changes added to commit."*

**Nothing was lost** — this page and the lesson are both in `HEAD`, verified by
content. **What was lost is attribution.** A 1,074-line rewrite of the
overseer's own current-state page is recorded in git as *"Review DAILY 10-06
acts 6-7: the T0.32 scope fork ruled (i)..."*, and `docs/LESSONS.md`'s 822+
entries gained one under the same message.

**Why this is a real defect and not a cosmetic one.** This project's cross-organ
discipline rests on being able to ask *"which organ did this, and in which
commit"* — it is the premise of `BUILDER-TRACE`, of the firing-transcription
diff, of `DELIVERED — AWAITING STAMP`, and of this organ's standing instruction
to **re-derive the fact a prior audit declined on**. The next audit looking for
the 143rd will not find it by message. And the defence that exists is
one-directional: *"stage by name"* protects the committing organ from swallowing
someone else's work; it does nothing when the OTHER organ stages with a
wildcard. The two organs collide at 06:37 by cron and the collision is known —
`cross-organ-doc-race-voids-certificates` was `ACTED` at `b4df9bb` — so this is
a surviving mode of an already-dispositioned class, which is the interesting
part.

**Routed to the builder as item 6.** I am not re-committing the content (it is
already correct in `HEAD`); this commit exists so the audit is findable by its
own message.
