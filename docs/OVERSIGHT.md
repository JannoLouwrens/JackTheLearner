# OVERSIGHT — 20th audit, 2026-08-20 00:45 UTC

## VERDICT: ON TRACK

Every hard integrity check is clean, the 19th audit's RANK 1 (the stale-PASS
cascade) is genuinely closed and verified by re-querying the detector, and the
day produced the best-constructed spec in the project's history (TA.02) plus a
new structural guard (SM.02's machine-enforced pilot-before-freeze). No silent
loosening in 7 days.

One new finding outranks all of that as a *mechanism* gap, though it has not yet
cost the ledger anything: **the `JACK_REUSE_KERNEL` reattach path can stamp a
certificate with code the experiment never ran, and no instrument in this
project can detect it.** It fired for the first time yesterday, on TA.02. I
diffed the laundered change and it is harmless. It was harmless because the
author was careful, not because anything stopped it.

---

## 0. Coverage — is the ladder the RIGHT ladder?

`python -m experiments.coverage` exits **0**. **Zero commitments with no
declared spec.** All 86 `COVERS` pairs carry an explicit kind; `declarations()`
returns an empty bad-list.

**15 of 23 commitments still have nothing passing** (down from 16 — taste
crossed yesterday). The important new decomposition, which no previous audit
has stated: of those 15, **13 have a claim-kind spec that is NOT IMPLEMENTED**,
and only 2 have been attempted and lost.

| commitment | claim-kind spec(s) | state |
|---|---|---|
| sight | OP.01, T3.01 | **not implemented** |
| smell | SM.02 | implemented 00:07 today, pilot in flight |
| voice | VO.02 | not implemented (needs a second Jack) |
| balance | BA.02 | **VOID** (D8: no actuator has catch authority) |
| proprioception | T3.02, UB.16 | not implemented |
| thermal (kills) | SH.01 | not implemented (rig parked at z=1.03) |
| shelter/building | SH.01 | same spec, same block |
| hunger/thirst | PS.04 | not implemented |
| sleep | ME.7, T5.05 | not implemented |
| death & retry | XL.01 | **FAIL** (attempt 2, variance-dominated) |
| social/other agents | T6.05, VO.02 | not implemented |
| plasticity | T5.03, T5.04 | not implemented |
| fast/slow | DP.01–DP.04 | not implemented |
| touch/contact | UB.5 | not implemented |
| tool use | CU.6 | not implemented |

This reframes the zero-pass number honestly. It is **not** a wall of failing
science. It is a wall of **unwritten tests** — the ladder is right and unbuilt.
The rate limit is builder throughput, not refutation.

**Concentration risk, new and worth naming.** `one brain / unison` — the
constitution itself — has **18 claim-kind specs and exactly one passing:
UB.9**. That same UB.9 is *also* `hearing`'s only passing claim. Two
constitutional commitments rest on one ledger row. If UB.9 ever stales or is
impeached, two commitments go to zero in the same instant. `curiosity` is nearly
as thin: 11 claim specs, one pass (T2.08).

`sight` deserves a sentence of its own. It is the first sense GOAL.md names,
it has five specs, and **neither of its two claim-kind specs (OP.01, T3.01) is
implemented**. Yesterday's fresh T2.03 PASS is credited a *fixture* — correctly
— so a real day of sight work moved the sight board by zero. That is the
coverage tool working exactly as designed, and it is telling us something.

---

## 1. Integrity of the ledger — CLEAN

82 PASS of 169 specs (90 ledger rows: 82 PASS, 4 FAIL, 4 VOID).

| check | result |
|---|---|
| PASS whose `commit` no longer exists in git | **0 / 82** |
| PASS with no implementation on disk (`module_path_for` strict) | **0 / 82** |
| PASS declaring a control whose `_check` never reads it (AST, 2nd formal) | **0 / 80** |
| PASS declaring no control at all | 2 — T0.01, T0.10, **by design** |
| PASS recorded from a dirty tree | **0** |
| PASS at tier ≥2 with fewer seeds than the spec declares | **0** |

The seven tier-2 single-seed PASSes (ME.1–4, PG.1–2, T2.00) all declare
`seeds=1` in the registry and are apparatus/existence claims, not learning
claims. No seed count was reduced.

`run verify` independently re-judges 81 PASS from the record alone (T0.18
self-excluded, correctly): 0 verdicts that no longer re-derive, 0 gates that
ignore their control, 0 controls declared but never run, 0 unreplayable, 0
unauditable, 0 / 0 undeclared-control budget consumed.

### 1.1 — The 19th audit's RANK 1 is CLOSED, and closed the right way

`run status` now reports **5 stale + 1 content-stale**, down from 10. Critically:

    T2.05  VOID    T3.07  FAIL    T4.02  FAIL
    LC.03  VOID    BA.02  VOID    T2.02  VOID (pre-impl_sha, content)

**Not one of them is a PASS.** Every stale *claim* is gone. `dd07693` and
`29b11b0` re-certified T0.12, T2.08, T2.03 and T2.04 on live code, and — the
part I asked for — `29b11b0`'s message quotes the detector's own output as the
proof of closure rather than a finished worklist. B1 discharged in full.

### 1.2 — RANK 1 (NEW): a reattach can stamp a certificate with code that did not run

**TA.02's PASS row records `commit 926541c`, `impl_sha f30e1ba6fa331f28`. The
kernel that produced every number in it — `jack-ladder-1787178802` — executed
the code at `40e6bad`, `impl_sha 2e7ec0969fa1e598`.**

The mechanism, exactly:

- `gpu.py:712` — `if not reuse:` guards `kernels push`. On a
  `JACK_REUSE_KERNEL` reattach the local script is written to `work/kernel.py`
  and **never pushed**. The remote code is whatever the *original* submission
  pushed.
- `run_spec` stamps `impl_sha` from the **local tree at recording time**.
- So any edit to the test file between dispatch and reattach-harvest is written
  into the certificate as though it had run.

`stale_claims()` compares *recorded* `impl_sha` against *current* — they match
perfectly, because the recorded one was taken from current. **The staleness
detector cannot fire on this by construction.** It was built to catch the
entry being about older code; this is the entry being about newer code, and
that direction has no instrument.

I found it by scanning all 90 rows for a `gpu_job_id` appearing in `history[]`
under a different `impl_sha`. **Exactly one hit in the project's history**, and
it is yesterday's.

**In this instance the science is intact and I want to say so plainly.** The
full diff `40e6bad..HEAD` on `ta_02_one_trial_aversion.py` is two lines:

    -from ..gpu import build_job, submit
    +from ..gpu import build_job, result_json, submit
    -    data = json.loads(r.artifacts["ta202.json"])
    +    data = result_json(r, "ta202.json")

Both inside `_submit`, the harvest path. No gate constant, no arm, no world
code, no `_check`. Every number in the row came from the kernel and the kernel
is unchanged. The builder also preserved the attempt-1 ERROR in `history[]`
with its true `impl_sha`, so a careful reader *can* reconstruct what happened —
the trail is honest. What does not exist is a machine that would have stopped a
less careful edit, and "the author was careful" is precisely the standard this
project exists to replace. Fix in **B1**.

---

## 2. Thresholds and controls over 7 days — NO SILENT LOOSENING

41 commits touched `registry.py`, `registry_expansion.py`, or `experiments/tests/`
in the last 7 days. I parsed every `NAME = <numeric>` assignment appearing on
**both** sides of every diff (a name only added is a new file, not a move).
**Exactly three commits move an existing constant, and none loosens:**

| commit | constant | move | verdict |
|---|---|---|---|
| `265e683` | `ALIEN_MIN_DIST` | 1.5 → 2.0 | **STRENGTHENED** — the author confessed 1.5 was the accommodation that disarmed the control |
| `a703604` | `N_PROPERTIES` | 8 → 9 | tightens |
| `1861b18` | BA.02's five PILOT-FINAL gates | **byte-identical** | comment text only |

Seeds held at 3 everywhere they were 3. No `_check` gained a disjunctive gate.
No assertion or `_control` was deleted — the only `-` line matching
`assert|_control|seeds` since the 19th audit is a `"reason"` string inside a
ledger provenance amendment.

**TA.02's gates were frozen before its run and never touched after.** All of
`POOL_AVOID_GATE 0.90`, `SEED_AVOID_FLOOR 0.80`, `POOL_DISC_GATE 0.60`,
`ACT_THRESH 0.002`, `SHOCK_CUE_GATE 0.90` were set at `40e6bad` (implementation)
and are unchanged at HEAD. Its `_check` is fail-closed throughout (`.get(k, 0.0)`
defaults), VOIDs on a dead rig or a null that never learned to eat, and FAILs —
not VOIDs — on a must-fail control that cleared, with the distinction reasoned
out in the code. This is the best-constructed gate in the ladder.

**No findings in section 2.**

---

## 3. Drift from the goal — ZERO drift; the converse is section 0

Five work items since the 19th audit. Each traces to a GOAL.md sentence:

| work | GOAL.md sentence |
|---|---|
| TA.02 implemented + PASS | *"gustation drives conditioned taste aversion, one-trial learning with long delay tolerance"* (l.47) + fast/slow axis 3 (l.247) |
| SM.02 implemented + pilot dispatched | *"olfaction finds food, fire and decay at a distance and through occlusion — the sense that works when sight fails"* (l.46) |
| T0.24 property P6 (AST guard) | *"protects the honesty of watching what happens"* (l.8) |
| T2.04 re-cert / T0.12 re-run | same — the certificate must be about live code |
| `dispatch.sh` GPU-lock pre-flight | same — a dispatcher that dies silently at exit 0 is a lying instrument |

Nothing serves nothing. **Zero drift, for the fifth consecutive audit.**

The converse question — which parts of GOAL.md have no passing spec — is
section 0, and its answer is now sharper than "16 zero-pass commitments": the
binding fact is that **13 of the 15 zero-pass commitments have never had their
claim spec written**, and four of those five families (fast/slow's DP.01–04,
plasticity's T5.03/04, sleep's ME.7/T5.05, curiosity's CU.1–7) are the exact
claims GOAL.md warns are *"most likely to be quietly neglected in favour of easy
wins."* They are not being neglected in favour of easy wins — the last two days
went to two of the hardest senses in the inventory — but they are not moving.

---

## 4. Is the builder alive and productive? — YES, best day in the project's history

18 iterations between 2026-08-19T07:31 and 2026-08-20T00:38 (17.1 h):

    rc=0   16
    rc=1    1   (11:07, 8 seconds — aborted immediately, no work lost)
    rc=124  1   (14:07→14:57, 50-minute timeout)

**PASS delta 80 → 82.** No paused loop, no credit exhaustion (session 15 %,
week 41 % at the 00:07 iteration), no repeated identical failure, no iteration
aborting on load (0.00–0.76 all day). The one timeout is the known pattern of a
detached run landing after its launching iteration ends, and the journal handoff
protocol caught it.

The two ledger-count oscillations in the log (`81 → 80` at 08:23, `80` at both
19:07 and 20:07) are the stale-cert churn, not lost claims — the re-certification
chain temporarily displaced rows it then re-recorded. Verified: no PASS is
missing from the ledger relative to the 19th audit.

Two structural improvements shipped, both meeting the "no new organ without a
scar" bar:

- **T0.24 P6** — a static AST scan of all 92 test files flagging `json.loads`
  applied to an `.artifacts` entry, with the pre-fix TA.02 line as a
  known-positive fixture. The scar: `result_json`'s docstring had documented
  this exact failure since 2026-08-11 and TA.02 hand-rolled the read anyway.
  The lesson the builder wrote — *a scar in a docstring is prose; only a check
  binds the next author* — is correct, and section 1.2 is the same lesson
  needing to be applied once more.
- **SM.02's `_GATES_FROZEN`** — `run()` hard-refuses the registered run until
  the pilot's numbers freeze the gates (`sm_02_smell_finds_occluded.py:615`,
  real code, not a comment). This is the 19th audit's B3 demand turned into a
  machine. First spec in the ladder to enforce pilot-before-freeze rather than
  trust discipline. **This is the single most valuable thing built yesterday**,
  more than either PASS.

---

## 5. Compute honesty — every W33 hour is accounted, one attribution hole

**W33 Kaggle: 2.0890 h billed OK + 0.0935 h billed failed = 2.1825 h of 30.
Remaining floor ≈ 27.81 h, expiring Sunday 2026-08-23 (~82 hours out).**

Every W33 hour maps to a ledger row:

| job | hours | produced |
|---|---|---|
| `1787124880` | 0.2970 | T2.06 PASS |
| `1787166872` | 0.0935 (failed) | T2.03 ERROR — torchvision pin stopped firing; diagnosed and fixed |
| `1787170366` | 0.3220 | T2.03 re-cert PASS |
| `1787173733` | 0.9599 | T2.04 re-cert PASS |
| `1787178802` | 0.5101 | TA.02 PASS |

**Zero wasted GPU-hours this week.** The one failed charge was correctly filed
as waste and produced the `_tv_v()` fix. The TA.02 recovery is a genuine win:
the reattach recorded a PASS for **0.00 h of new quota** with idempotent billing
(W33 unchanged across the second attempt — verified, `charged_jobs` holds one
entry for `1787178802`).

**RANK 3 finding — pilot GPU spend is unattributed.** `spec` in
`gpu_submissions.jsonl` comes from `JACK_SPEC_ID`, which `run_spec` sets around
the seed loop (`gpu.py:908,917`). Pilots call `submit` **outside** `run_spec`,
so they file with `"spec": ""`. **27 of 49 receipts are unattributed**,
including the in-flight SM.02 pilot at `est_hours 2.0` — roughly 7 % of the
remaining W33 quota, filed against no spec.

Accounting is not affected: `gpu_budget.json` is keyed by `job_id` and its
weekly totals are correct. **Attribution** is what is missing, and it matters
*more* now than it did a week ago, because pilot-before-freeze just became
machine-enforced (SM.02) and a standing lesson twice over. Pilot spend is
becoming structural rather than incidental, and a per-spec cost audit currently
cannot see any of it.

**W31 still reads 37.4554 kaggle-hours against a 30.0 ceiling with
`overruns: []`** — the 19th audit's B5, backfilled in one commit so the
`charge()` path never fired. **Not addressed**; carried forward as B4. Low
severity, no claim depends on it, but it is a guard with an unnamed condition.

---

## 6. Stuck decisions

**D1 (does the 57M trunk stay in the control path?) is open 11 days and is
blocking the ladder's central spec.** T2.01 — *Locomotion beats a random
policy* — has read `FAIL` since 2026-08-12T12:59 with no attempt 3 in eight
days (history: VOID 08-07, FAIL 08-10, FAIL 08-12). It is `GPU_LONG`, it billed
5.58 h on each of its two runs, and it cannot be re-dispatched without the
one-line fork D1 asks for.

This is worth stating in GOAL.md's own terms: **the ladder-and-apple standard
requires him to climb, and locomotion is the spec that says he can move at
all.** It is the most consequential FAIL on the board and it is blocked on a
question the owner has not been asked in six days. The fork is unchanged and
still one line:

  (i) strike option A (freeze the trunk) — PLASTIC-ONLY stands as written; or
  (ii) keep option A and narrow the decree's scope, saying where.

**Nothing was quietly acted on.** I checked: no owner-decision in
DECISIONS_NEEDED has been resolved in code without being recorded. The
credits-policy entry's 19th-audit addendum correctly stands down its own
urgency for this week rather than letting a stale premise drive a decision —
that is the right behaviour and I am noting it as such.

**Nothing blocked is resolvable by a bakeoff the system could have run itself.**
D1 is a constitutional question (does the decree admit a frozen tower), not a
measurement question; the 15th audit was right that "do what the measurements
say" cannot answer it.

---

## 7. Bakeoff hygiene — CLEAN

Three entries in `DECISIONS_RESOLVED.md`:

- **PS.01/J — VOID.** Three arms below the 3.0σ learning gate. Correctly
  recorded as VOID and *not* treated as a verdict. This is the learning gate
  doing exactly its job.
- **PS.01/J2 — WINNER, `impact_speed`.** Re-run after fixing the arms.
- **D2 — WINNER, VOID BLOCKS its dependents**, resolved by *ledger replay*
  rather than argument, with the counterfactual measured (specs admitted at
  01:00 under each semantics) and made executable as T0.08 property 6, with a
  named re-open trigger.

No decision made without a learning gate. No VOID treated as a verdict. No
winner chosen inside a noise margin. **No findings in section 7.**

---

## 8. The honest summary

**Yesterday we got closer to Jack, and not only to a longer list of green ticks.
That has not been true on most days I have audited.**

The case for it: taste is now a *capability*, not a spec — one-trial
conditioned aversion at 0.983 pooled avoidance, surviving death on every seed,
with the Garcia double dissociation clean in both directions (poison → taste-
selective, shock → cue-selective) and a DQN given the same information eating
196–218 toxic meals across 150 lives at zero discrimination. That is a learning
mode nothing else in Jack's design has, it is the fastest learning in biology,
and it is the first of the constitutional zero-pass senses to fall. Smell's
apparatus was probed before its test was written — LOS-visible 0.000 behind the
shelters and 0.87–0.91 with them removed, with a first layout rejected on
measurement for leaking vision at 0.43. That is the world being made honest
before the claim is made about it.

The case against complacency, and it is the one I want on the record: **13 of
the 15 zero-pass constitutional commitments have never had their claim spec
written**, and the ones that stay unwritten are the ones GOAL.md flagged as most
likely to be neglected — fast/slow (0 of 4 implemented), plasticity (0 of 2),
sleep (0 of 2), curiosity (1 of 11), touch (0 of 1). `one brain / unison`, the
constitution itself, rests on a single ledger row that is simultaneously
hearing's only row. At two claims per day against 13 unwritten claim specs plus
the senses that need a second Jack, the ladder-and-apple demonstration is not
weeks away.

And the thing that has not moved at all: **he still cannot be shown to walk.**
T2.01 has been FAIL for eight days, blocked on an owner question eleven days
old. Every sense we certify is a sense belonging to a creature that has no
demonstrated locomotion. That asymmetry — senses accumulating while the body
stays unproven — is the shape this project would take if it were drifting
toward what is easy to measure, and it is the thing I will be watching hardest
next audit. It is not drift *yet*, because the block is external and honestly
recorded. It becomes drift the day the builder stops noticing.

The machine itself got measurably better: the stale-PASS cascade closed and
proved closed by its own detector, a docstring scar became an executable check
across 92 files, and pilot-before-freeze stopped being a discipline and became a
refusal in code. Set against that, section 1.2 is the machine's one blind spot
found this week — and finding it costs nothing to fix.

---

## FOR THE BUILDER

**B1 (RANK 1) — make a reattach unable to launder a code edit into a
certificate.**

`JACK_REUSE_KERNEL` skips `kernels push` (`gpu.py:712`), so the remote code is
the *original* submission's, while `run_spec` stamps `impl_sha` from the local
tree at recording time. TA.02's PASS therefore names `impl_sha f30e1ba6…` for
numbers produced by `2e7ec096…`. `stale_claims()` compares recorded-vs-current
and so can never fire on this direction.

Verified harmless *this* time — the whole diff is two lines inside `_submit`
(the harvest path), no gate or arm touched — so **no verdict is impeached and
nothing needs re-running.** Fix the mechanism, not the row.

Cheapest sufficient fix: `submit` already writes `work/kernel.py` before push.
Record `sha256(kernel.py)` in the **attempt** receipt at push time. On reattach,
recompute it from the local script and compare; on mismatch, either refuse to
record, or record and force an entry in `amended[]` naming both shas. Either is
fine — what is not fine is the certificate silently claiming the newer code.

Belt-and-braces, if it is cheap: have `run_spec` carry the reattached job's
original `impl_sha` into the row when it differs, so the ledger states the fact
rather than relying on a reader reconstructing it from `history[]`.

The generalisable half, for LESSONS: **staleness was built to catch a
certificate about OLDER code, and it has no instrument for a certificate about
NEWER code.** Any path that decouples *when the code ran* from *when the row was
written* — reattach today, a resumed checkpoint or a cached artifact tomorrow —
reopens this hole. The invariant is not "the sha is current", it is "the sha is
the sha of what executed".

**B2 (RANK 2) — attribute pilot GPU spend.**

27 of 49 receipts in `gpu_submissions.jsonl` carry `"spec": ""`, because
`JACK_SPEC_ID` is set by `run_spec` and pilots run outside it — including the
SM.02 pilot now in flight at `est_hours 2.0`, ~7 % of the remaining W33 quota.
Weekly accounting is correct (charges are job-keyed); *attribution* is not, and
a per-spec cost audit is currently blind to every pilot hour.

Fix is one line at each pilot call site: export `JACK_SPEC_ID` (or pass an
explicit `spec=` through to `_receipt`) before calling `submit`, and add
`phase: "pilot"` so pilot spend can be summed separately from registered spend.
Worth doing now rather than later, because `_GATES_FROZEN` just made piloting
mandatory-by-machine and pilot hours are about to become a standing fraction of
the budget. When SM.02's pilot lands, backfill its receipt's `spec` field.

**B3 — sight has five specs and neither of its two claim specs exists.**

`OP.01` and `T3.01` are the only claim-kind specs credited to `sight`, and both
read `not implemented`. T2.03 is a fixture and correctly buys sight nothing.
Sight is the first sense GOAL.md names. `OP.01` (*"A thing behind the rail still
exists"* — object permanence) is tier 2 and looks the cheaper of the two; it is
also the natural companion to SM.02, which is currently proving what smell can
do *when sight fails*. Consider it the next non-GPU pick.

**B4 — make `overruns` mean what it says** (carried from the 19th audit's B5,
unaddressed).
W31 holds 37.4554 kaggle-hours against `KAGGLE_WEEKLY_HOURS = 30.0` with
`overruns: []`, because the figure was backfilled rather than charged — the
ceiling check lives only in `charge()` (`gpu.py:455-461`). Either run it on
reconciliation/migration writes too, or have `Budget` report the breach at read
time. Low severity; no claim depends on it; but it is a guard with an unnamed
condition and it has now survived two audits.

**B5 — XL.01 attempt 3 still needs its power calculation first** (carried, and
correctly respected — no attempt 3 was launched). The 19th audit's numbers
stand: identical v2 fixture, ratio 0.084 on worlds 0–2 and 1.003 on worlds 3–5,
`search_time_ratio_std` 0.671 on a mean of 1.003. Pilot the carried/wiped
contrast on 6–8 worlds at one seed each, compute the between-world std, and size
`N_LIVES` and `spec.seeds` from it before re-registering. If the required N does
not fit `CPU_LONG`, say so and escalate.

**B6 — spend the expiring quota.**
27.81 h of W33 Kaggle dies **Sunday 2026-08-23**, ~82 hours out; credits are
healthy (session 15 %, week 41 %). SM.02's pilot claims ~2 h of it and its
registered run will claim more. The remaining GPU-shaped zero-pass picks are
VO.02 (heavy — needs a second Jack) and, if D1 ever clears, T2.01 at 5.58 h.
On present course W33 closes with roughly 20 h unspent. That is a scheduling
choice, not a block — but say so explicitly in a commit message if it is the
choice being made, so the next audit is not reading it as neglect.

---

## FOR THE OWNER

**One question, and it is the same one, now eleven days old and blocking the
spec that says Jack can move.**

`docs/DECISIONS_NEEDED.md` **D1** — *does the 57M trunk stay in the control
path?* — needs one of two answers:

  **(i)** strike option A (freeze the trunk); the PLASTIC-ONLY decree of
  2026-08-09 (`GOAL.md:76`) stands as written; or
  **(ii)** keep option A available and narrow the decree's scope, saying where.

Please **do not** answer *"do what the measurements say"* — a trigger written
earlier in that file would read it as option A, and the 15th audit established
that this is a constitutional question about what the decree admits, not a
measurement question a bakeoff can settle.

What it costs while it waits, updated: **T2.01 (*Locomotion beats a random
policy*) has read FAIL since 2026-08-12 with no attempt in eight days.** It
cannot be re-dispatched without this fork. It is the precondition for the
ladder-and-apple standard in GOAL.md — every sense certified since is a sense
belonging to a creature with no demonstrated locomotion. 27.8 free GPU-hours
expire Sunday and T2.01 needs 5.58 of them; it fits five times over.

*Raised without taking a side.*

**Also on your desk, unchanged and NOT urgent this week:** the standing policy
for when the weekly credit meter exhausts while a GPU quota expires unused
(options (a) accept the rate cap / (b) reserve N % of the meter for GPU-dispatch
iterations / (c) raise the ceiling by exception in an expiry week). It has
happened five times. It is **not** happening this week — credits are at 41 %
weekly and the loop ran 18 iterations yesterday. Answer it when convenient; I
still do not recommend one.

---

*20th audit. Read-only: no spec, test, model, gate, threshold or ledger row was
modified. `experiments/ledger.json` untouched. This file and any
`DECISIONS_NEEDED.md` / `LESSONS.md` appends are the only writes.*
