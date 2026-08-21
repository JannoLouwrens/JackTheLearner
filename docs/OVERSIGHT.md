# OVERSIGHT — 24th audit, 2026-08-21 00:40 UTC

## VERDICT: ON TRACK — every integrity check is clean, and the escalation on my desk resolves AGAINST the builder's diagnosis but IN FAVOUR of the gate that caused it: T3.01's shuffled control is not a dead rig, it is a live arm sitting on the max-entropy fixed point, and the 23rd audit's B1 gate was right to void the PASS it retracted

The hard checks are the cleanest they have ever been. 82 PASS, every one naming
a live commit, resolving to an implementation, and declaring a control that its
gate provably reads; `run verify` re-judges 81 from the record alone with zero
failures on all five probes, and the undeclared-control debt — a ratchet the
overseer watched climb 19 → 20 across two audits in August — now reads **0 / 0**.
**Zero module constants moved in any pre-existing spec or test in seven days**,
verified by parsing the AST at HEAD against the tree at `ea2bdbf`. No stale PASS
exists; all three stale entries are VOID.

The day's headline is not a new capability. It is that the machine caught the
same class of bug twice in thirty hours, in two different specs, and paid for it
by retracting its own newest and most-wanted claim:

> **A control that never left the uniform predictor reads exactly chance —
> whether or not the rig leaks.** On 2026-08-20 at 15:29 T3.01 recorded sight's
> first claim-kind PASS. Its leak control read `acc_shuffled` [0.25, 0.2433,
> 0.25], `shuffled_dev_max` 0.0067 — dead-on chance, apparently clean. The
> 23rd audit's B1 gate then required that control to demonstrate it had trained
> at all, and on its first real exercise it read `shuffled_fit_min` 0.25 and
> **VOIDed the claim at 20:26**. Sight went back to 0-pass and the ladder went
> 83 → 82. That retraction is the single most valuable thing in this audit.

---

## The adjudication the builder is blocked on — T3.01, probe branch R3

`experiments/t301_shuffle_probe.py` pre-registered R0–R3 before launch, fired
**R3 (ESCALATE)**, and froze T3.01 as undispatchable pending this audit
(commit `4b9ff30`). R3's branch text reads: *"some seed has NO live lr at all.
That contradicts the main arm's demonstrated learning under identical code and
optimiser — a deeper rig fault."*

**Ruling: the trigger fired correctly; the conclusion attached to it does not
follow. There is no rig fault.** Three readings from the probe's own artifact
(`/data/t301_shuffle_probe.json`, 9 rows, 124 epochs, P100):

**1. It is not true that no seed has a live lr.** At **lr 1e-4 all three seeds
are escaping the plateau** by the end of the run — loss falls 1.3866→1.3769
(seed 0), 1.3872→1.3705 (seed 1), 1.3871→1.3745 (seed 2), and shuffled-train
fit climbs off a flat 0.2500 to 0.3175 / 0.2875 / 0.2800. "No live lr" is an
artefact of the `LIVE` predicate being pinned at *0.05 loss fall **by epoch
62***, a magnitude and a horizon both chosen a priori and never calibrated.

**2. The premise "contradicts the main arm under identical code" is false.**
The two arms differ in the one way that governs optimisation: the main arm's
labels carry structure, the control's are random. On random labels the uniform
predictor is a genuine attractor at loss = ln 4 = **1.386294**, and every "dead"
row sits on it to four decimals (1.3863–1.3880) with fit at *exactly* 0.2500 on
a set the shuffle keeps balanced 300/class. Sitting at the max-entropy fixed
point is what a correctly functioning network does on random labels **before
memorisation onset** — it is the probe docstring's own Zhang-et-al. premise
coming true, not a broken rig.

**3. The "sharpest anomaly" is not anomalous.** The builder read seed 1 vs
seeds 0/2 *within* lr 3e-4 and found it unexplained. Read *across* the lr axis
the grid is orderly:

| lr | seed 0 | seed 1 | seed 2 | reading |
|---|---|---|---|---|
| 1e-4 | 0.2500 → 0.3175 | 0.2575 → 0.2875 | 0.2708 → 0.2800 | all three escaping, slowly |
| 3e-4 | pinned 0.2500 | 0.3108 → **0.5367** | pinned 0.2500 | the boundary |
| 1e-3 | pinned 0.2500 | pinned 0.2500 | pinned 0.2500 | all three pinned |

That is a single plateau-escape threshold between 1e-4 and 1e-3, with 3e-4
straddling it and escape being seed-dependent at the boundary. The probe varies
seed precisely so this is visible.

**So the substantive finding is R2's** — *"the floor sits above what a live
matched-budget arm reads on random labels"*, now measured rather than argued.
**R2 could not fire because R2's own trigger is gated on the same epoch-62
horizon that R2 exists to indict.** The rule was self-defeating: when the
horizon is the thing that is wrong, the only reachable branch is "deeper rig
fault." That is a generalisable lesson and it is appended to `LESSONS.md`.

**But R2's repair as written is NOT licensed, and this is the part that must
not be skipped.** `SHUFFLE_FIT_FLOOR` exists because the 15:29 PASS rested on a
control that had never left the uniform predictor on 2 of 3 seeds, so its
at-chance *test* reading was not evidence of no-leak. Swapping the fit floor for
a loss-fall observable at a recalibrated horizon re-labels *"the control moved a
little"* as liveness, when what a leak test needs is *"the control was ABLE to
exploit a leak if one existed"* — i.e. memorisation capacity. The probe prices
that: 0.5367 after 124 epochs on the single best row, and below 0.32 everywhere
else. A loss-fall proxy at any horizon inside that regime has no calibrated
relationship to leak-detection power. It would be the UB.10 disease with a new
observable.

**Reading the code changed my recommendation, so it is stated separately.**
T3.01 does not draw train and test from a shared pool — it generates them from
independent seed streams (`t3_01_ablate_vision.py:234-235`, `_build_dataset(seed,
n_train)` vs `_build_dataset(seed + 500_009, n_test)`). The hazard the shuffled
control is documented to detect ("the rig leaks episode identity") is therefore
a **deterministic property of two generators**, provable in milliseconds on CPU
by hashing 1200 + 300 frames and asserting an empty intersection. The project
has now spent **0.70 GPU-hours across three kernels and four loop iterations**
inferring stochastically what one `O(n)` assertion can prove. Honest caveat, so
this is not read as licence to drop the shuffled arm: frame-hash disjointness
catches identity contamination and **not** a distributional leak (a per-class
rendering artefact reproduced in both streams would pass it and still leak) —
which is exactly what a *live* shuffled arm catches. They are complements.

T3.01 is **un-frozen for redesign**. It is not cleared to re-run as-is.

---

## 0. Is the ladder the right ladder?

`experiments.coverage` exits 0. **0 of 23 constitutional commitments has no
declared spec** — the 2026-08-10 miss stays closed for the eleventh straight
audit. But the second number is the one that matters and it moved the wrong way
today:

**8 of 23 commitments have a claim-kind PASS; 15 have none.** Sight lost its
only one at 20:26. Six commitments show a passing spec that the tool refuses to
credit because it is a fixture or a sensor, not a claim (smell/SM.01,
voice/VO.01, balance/BA.01, thermal/PS.02, hunger/PS.01, death/XL.00) — the
world can do the thing; nobody has yet shown *Jack* does.

The two that GOAL.md warns are most likely to be quietly neglected are exactly
the two that are:

- **Curiosity — 12 specs, 1 claim PASS.** T2.08 ("Curiosity drives coverage").
  T5.06 ("Unprompted exploration is real") — the ladder-and-apple sentence
  itself — is NOT_RUN, as is the whole CU.1–CU.7 family. **T2.09, the noisy-TV
  control spec written to adversarially test T2.08's own claim, has never run**,
  though PG.4 proves the trap exists in the world. T2.08's in-spec control (a
  time-permuted magnitude-matched bonus) is real and is read; the spec-level
  adversary is not.
- **One brain / unison — 21 specs, 1 claim PASS.** UB.9. T4.02 ("No modality
  collapse") is FAIL at 27.28× vs a 10× gate. UB.7, the headline UNISON claim,
  is NOT_RUN. UB.10, the fusion bakeoff meant to move this, is parked.

## 1. Integrity of the ledger — no findings

| check | result |
|---|---|
| PASS resolving to an implementation in `experiments/tests/` | 82 / 82 |
| PASS whose `commit` still exists in git | 82 / 82 |
| PASS recorded from a dirty tree | 0 |
| PASS declaring a control with no `control_metrics` | 0 |
| PASS with fewer seeds than its spec declares | 0 |
| `run verify` — verdicts that no longer re-derive | 0 / 81 |
| `run verify` — gates that IGNORE their control (probe B) | 0 / 79 |
| `run verify` — gates unreplayable / entries unauditable | 0 / 0 |
| controls run but NOT declared (the ratchet debt) | **0 / 0** — paid off in full |
| stale PASS claims | **0** (all 3 stale entries are VOID: LC.03, BA.02, T2.02) |

Two PASSes declare no control at all — T0.01 and T0.10 — unchanged and by
design, and still the honest caveat recorded in earlier audits: an existence
claim whose gate was never shown capable of reporting the bad case.

Six entries carry an `amended` record (T0.05, T0.09, T1.07–T1.10). Every one is
**provenance-only** (a `hardware` string, an `attempt` count), attributed to a
named audit, with a reason. **No metric and no verdict has ever been amended.**

T3.01's retraction is recorded the way the 22nd audit's guard demands:
`supersedes_void` carries the prior commit, `impl_sha`, `dirty` flag,
`impl_changed: true` and the full failing metric block. The audit trail of a
claim being taken away is complete.

## 2. Thresholds and controls over time — no findings, third clean week

32 commits touched `registry.py` / `registry_expansion.py` / `tests/` in seven
days; 15 in the last 24 h.

- **0 module-level constants moved** in any file that existed at `ea2bdbf`
  (AST parse of both trees, not a diff read).
- **0 spec or test files deleted.**
- **17 deleted lines total** across `experiments/tests/` in seven days. Not one
  is an assertion, a threshold, a seed count, or a control. (They are a call
  signature and a `properties_checked` count that went *up*.)
- Every registry edit is **strengthen-only**, and each names its measurement:
  T2.04 and T2.05 **gained** controls before their first run; T2.05's null was
  replaced by `min(persistence, mean)` because the 08-14 VOID *measured*
  persistence uninformative at K=5; T3.01 gained its declared control after
  `UndeclaredControl` fired; T0.24 and T0.27 gained obligations.
- No `_check` gained an `or`.

The one movement worth naming is in the tightening direction and cost a PASS:
`SHUFFLE_FIT_FLOOR` was added, fired on its first real exercise, and took sight
back to zero. That is law 4 working in the direction law 4 is hardest to obey.

## 3. Drift from the goal — no drift; the risk is the opposite

Everything the builder touched in 24 h traces to a GOAL.md sentence:

| work | GOAL.md sentence it serves |
|---|---|
| T3.01 ×3 runs + 2 probes | *"ablate a sense, something measurable must degrade"* (sight) |
| T2.05 redesign + run | *"world model"* / fast-slow acting; priced the LC.04 arms |
| T4.02 re-run | *"no modality collapse"* — one brain, senses in unison |
| T3.07 re-run | *"components must EARN their parameters via ablation or be deleted"* |
| UB.10 pilot + recipe probe | *"a genuinely unified brain where every sense is load-bearing"* |
| LC.03 registered run (in flight) | *"he lives, he dies, he remembers"* — survival core |
| W0.BAL → D9 | the body; T2.01's *"needs a better body"* |
| Provenance guards (T0.24/T0.27) | *"protects the honesty of watching what happens"* |

**No drift.** The converse is the finding, and it is section 0's: the loop is
working hard on the instrument and the north-star claim is untouched. Three of
the four capability verdicts recorded in the last 24 h (T2.05, T3.07, T4.02)
were **predicted in writing before the run** — excellent anti-run-until-pass
hygiene, and also a statement that ~3.45 GPU-hours bought confirmation of things
already known. The one run that could have surprised was retracted.

## 4. Is the builder alive and productive? — alive, disciplined, and running out of runway

- **26 iterations in 24 h, all `rc=0`.** The last non-zero exit was 2026-08-19.
- **PASS delta: 83 → 82 (−1).** Honest: the loss is the T3.01 retraction.
- LC.03's registered run is **verified alive by process, not by claim** (the
  audit-check-the-builder-is-not-idling rule): pids 92859/92860/92861, elapsed
  14 h 18 m, CPU time 14 h 06 m–14 h 40 m, 98–102 % CPU. It writes the ledger
  itself and lands ~01:20 UTC.
- **Two exhaustion clocks, both close.** The Fable weekly meter reads **85 %
  against a 90 % hard stop** (resets 2026-08-24); the builder has correctly gone
  lean, but a hard stop inside 5 points is a foreseeable pause nobody has a plan
  for. And 22.81 Kaggle hours expire the same day.
- **The queue is empty and that is the real problem.** By the builder's own
  count every GPU-capable spec is unavailable: SM.02 parked, UB.10 parked,
  T3.01 escalated (unfrozen by this audit), and everything behind T2.01/D1.

## 5. Compute honesty — no findings

W33: **7.1877 h charged** (6.9299 ok + 0.2578 failed), **22.81 h remaining**,
resets Sunday 2026-08-24.

Last 24 h: **~3.45 h across 13 kernels.** Every one is attributable to a named
job with a recorded outcome, and every kernel that produced a registered run
produced a ledger row — T2.05 FAIL, T3.01 VOID, T3.01 PASS, T4.02 FAIL, T3.01
VOID. The six kernels with no ledger row are pilots and diagnostic probes (two
T3.01 pilots/probes, the curves probe, two UB.10 probes, one dead import), which
is by design and each is named in a commit message. **No unattributed hours.**

Three kernels died on environment faults for 0.164 h total (mujoco sdist-vs-wheel
×2, the hardcoded chdir ×1) — small, and each produced a LESSONS entry.

The honest accounting note: 3.45 GPU-hours produced **zero net new capability
claims and one retraction**. That is not waste in the ledger sense; it is the
price of a day spent auditing the instrument, and I would rather pay it than
not. But it cannot be many days in a row, and 22.81 free hours expiring into an
empty queue in under three days is the cost of the owner queue below.

## 6. Stuck decisions — one finding

- **D1** (does the 57M trunk stay in the control path) — open **12 days**. It
  blocks T2.01, which blocks 36 specs including the only claim specs for six of
  the owner's own senses. Prior audits have priced it four times; I am not
  re-arguing it, only recording the new fact in §FOR THE OWNER.
- **D7** (MovementMoodCoupling) — declared READY TO DECIDE by the 23rd audit and
  confirmed by measurement since: T3.07 re-ran on current code and came back
  **bit-identical** to the 08-13 row, proving the two IMPL_DEPS drifts do not
  touch the mood→action path. There is no experiment left that would inform it.
- **D9** (the body fork) — newly raised, well-formed, W0.BAL's pre-registration
  preserved verbatim, and the stale "LC.03 cannot mean anything until this is
  decided" premise explicitly corrected rather than quietly dropped. Good work.
- **Nothing was quietly acted on.** I checked the converse directly: no body or
  world-geometry change landed in seven days. `playground.py`'s only edit adds
  SH.01's shelter geoms behind a default-empty argument, documented as
  byte-identical when unused; `survival.py` adds two accounting keys. Neither
  touches the rover, which is D8/D9's subject matter.

**The finding — D9 escalates permission to MEASURE, not permission to ACT.**
The W0.BAL bakeoff is CPU-only, minutes long, on four idle cores, and its
option (b) asks the owner to *order it run* so that they can "pick A/B/C with
evidence instead of taste." The builder is correct that **adopting** any winner
is owner-gated (D8 settled that body changes are world-contract changes). It
does not follow that **running** it is: D8's own four scratch probes were run at
this desk without owner permission on 2026-08-14, which is the precedent. Law 3
says decisions are made by bakeoff; adoption is a separate act. Routing a
free measurement through the slowest resource in the system — an authority with
a decision open 12 days — converts minutes of CPU into an indefinite block, and
guarantees the owner answers D9 with less evidence than they could have had.

## 7. Bakeoff hygiene — no findings

Three records in `DECISIONS_RESOLVED.md`, all sound:

- **PS.01/J — VOID, and treated as a VOID**: three arms below the 3.0σ learning
  gate, no winner declared. A VOID correctly not read as a verdict.
- **PS.01/J2 — WINNER `impact_speed`**: 10.32σ over the null, **2.66σ over the
  runner-up** — outside the noise margin, not inside it. Eleven arms eliminated
  by the gate are recorded by name. `screen` gate-mode is used, and its
  rationale is written down and correct: the arms are deterministic reductions
  of one memoised set of rollouts, so there is no training that could have
  failed and T2.02's broken-run-or-worse-architecture ambiguity does not exist.
- **D2 — resolved by ledger replay, not `run_bakeoff`**, with the departure from
  law 3 justified in the record (two readings of a dependency graph, no seeds,
  no null, nothing that could have failed), the loser recorded, and a re-open
  trigger attached to the quantity the decision rests on.

Both bakeoffs carry a learning gate, and the one that failed its gate returned
VOID instead of a confident wrong answer. The third law's exercise record is
thin but honest.

## 8. The honest summary — are we closer to a curious humanoid?

**In capability: no. The ladder is one rung shorter than yesterday and sight is
back to zero.**

**In trustworthiness: materially yes, and it is the more valuable of the two
today.** Inside thirty hours the system found the same disease in two unrelated
specs — a null that reads perfectly clean because it never trained — named it in
UB.10, generalised it into a gate for T3.01, and then let that gate take away
sight's first claim five hours after it was won. Nobody argued. The builder then
declined to repair the gate to taste, spent 0.37 h buying the calibration data
the gate never had, and escalated a branch that pointed at its own work. That is
the loop running on itself exactly as SYSTEM.md describes, and a project where
83 is allowed to become 82 is a project whose 82 means something.

What I cannot say is that we are closer to the ladder and the apple. GOAL.md's
defining image — climbing on attempt 40 after falling on 1–39, unprompted —
sits behind T5.06, which has never run, in a commitment holding 1 claim-kind
PASS out of 12 specs. Twelve days of daily audits and that number has not moved.
The reason is legible and is not the builder's fault: the body cannot stand
(`upright_cos` −0.041), locomotion is stuck at 2.67σ against a 5σ bar, and the
decision that would unblock both has been on the owner's desk since 09 August.
The machine is in excellent health and is running out of things it is allowed to
do. That is the shape of this audit, and it will be the shape of tomorrow's
unless D1 or D9 moves.

---

## FOR THE BUILDER

Ranked by damage to the trustworthiness of the ledger. B1 unblocks a frozen
spec and 22.81 expiring GPU hours; do it first.

**B1 — T3.01 is UN-FROZEN. The R3 escalation is adjudicated: there is no rig
fault.** Record this adjudication in the probe file's PROBE RESULT block (append,
do not rewrite the R3 finding) and cite it. Specifically:
  - R3's trigger fired correctly; its attached conclusion ("deeper rig fault")
    is refuted by the probe's own tail. All three seeds escape the plateau at
    lr 1e-4 by epoch 124; the flat rows sit on the ln 4 = 1.386294 max-entropy
    fixed point, which is the *correct* pre-memorisation behaviour on random
    labels, not a dead rig.
  - The seed-1-vs-seeds-0/2 "anomaly" dissolves when the grid is read across
    the lr axis rather than within one lr: a single escape threshold between
    1e-4 and 1e-3, with 3e-4 straddling it.
  - The substantive finding is **R2's diagnosis**: the 0.35 floor sits above
    what a live matched-budget arm reads on random labels at epoch 62. Measured.
  - **R2's repair as written is not sufficient** — see B2.

**B2 — before re-running T3.01, add the deterministic check and pre-register the
fate of the stochastic one.** Two parts, in this order:
  - (a) **Add a structural leak gate that cannot sit on a plateau.** T3.01
    generates train and test from independent seed streams
    (`t3_01_ablate_vision.py:234-235`). Hash all 1200 train frames and all 300
    test frames and assert an empty intersection; VOID on any collision. `O(n)`,
    CPU, milliseconds, deterministic, no GPU. This *proves* the hazard the
    shuffled control is documented to infer.
  - (b) **Then pre-register, before seeing any new number, which of two fates
    the shuffled arm takes** — this is a fork you commit, not a choice you make
    after the run: **(i)** budget it to demonstrated memorisation (the probe
    prices this: >124 epochs at a control-selected lr, and 2 of 3 seeds do not
    get there at all, so the honest cost is several times the claim's budget),
    with `SHUFFLE_FIT_FLOOR` retained; or **(ii)** demote it from a VOID gate to
    a recorded diagnostic, with (a) carrying the gate. Do not adopt a loss-fall
    liveness proxy at a recalibrated horizon: it re-labels "moved a little" as
    liveness and reintroduces the exact ambiguity the gate exists to remove.
  - Honest caveat to carry into whichever you pick: (a) catches identity
    contamination and **not** a distributional leak (a per-class rendering
    artefact reproduced in both streams passes (a) and still leaks). (a) and a
    *live* shuffled arm are complements. Say so in the docstring.

**B3 — audit the OTHER at-chance controls for the same disease, before it costs
a third spec.** UB.10 and T3.01 both shipped a control whose clean reading was
indistinguishable from a dead arm. That is now twice. Grep the ladder for gates
whose PASS condition is *"the control sits at chance"* and, for each, answer in
one line in the docstring: **what observable distinguishes "converged and found
nothing" from "never trained"?** Where there is no answer, that is a finding —
record it; do not silently add a gate. This is a cheap CPU sweep and it is the
kind of work that makes a bug unrepeatable rather than fixed.

**B4 — run the W0.BAL bakeoff on CPU and attach the numbers to D9. Adopt
nothing.** Running is not adopting. Precedent: D8's four scratch probes were run
at this desk on 2026-08-14 without owner permission, and they are the strongest
evidence D9 carries. Produce `upright_frac` and `hand_reach_z_max` for arms A/B/C
against the already-measured null (−0.041), under the pre-registration preserved
verbatim in D9, and append the table to the D9 entry. Then leave A/B/C on the
owner's desk exactly as it is. If the kill criterion fires (no arm gets a hand
above the first rung), that is itself a finding the owner needs before choosing.
Cost: minutes, four idle cores, zero GPU.

**B5 — LC.03 lands ~01:20 UTC.** Harvest it under the 23rd audit's standing note
(its control (e) is a rig tripwire, not a must-fail) and record the verdict
whatever it is. It is the only capability result in flight.

**B6 — name a plan for the Fable hard stop.** The meter reads 85 % against a
90 % stop that resets 2026-08-24. Write into the journal what the loop does if
it hits the stop with LC.03 harvested and D1 still open — a foreseeable pause
with no stated behaviour is how 4d18h of dead loop happened on 2026-08-19.

## FOR THE OWNER

Nothing here is new science and I am not re-arguing any decision. Three facts
that changed, and one number:

**1. D7 has been ready for 30 hours and takes ten seconds.**
MovementMoodCoupling failed its ablation twice, and yesterday's re-run on
current code came back **bit-identical** — proving the two code drifts since the
original verdict do not touch the mood→action path. There is no measurement left
that would inform it. Delete it, redesign it, or **accept it on the record as
cosmetics** — (c) is a legitimate answer and closes it. Until then the model
carries a component that has twice failed to earn its parameters, which is the
disease `GOAL.md:87` exists to prevent.

**2. D9 (the body fork) is new and is the first time three independent
measurements sit side by side**: he topples (`upright_cos` −0.041), he cannot
catch himself (D8's probes: no actuator has directional catch authority), and he
does not really locomote (T2.01, 2.67σ against a 5σ bar, curve converged). I
have told the builder to run the bakeoff and bring you the numbers **without
adopting anything**, so that D9 reaches you with evidence rather than three
options and a shrug. Your decision is unchanged: which lane (park / order /
fold into D1). You will just have the table when you make it.

**3. The number: 22.81 free Kaggle GPU hours expire on Sunday 2026-08-24, into
an empty queue.** Not because the loop is idle — it ran 26 iterations in 24 h,
all clean — but because every GPU-capable spec is parked, escalated, or behind
**D1, open since 09 August (12 days)**. I have unfrozen one of the four today
(T3.01), which may absorb some of it. The rest will expire. I am recording the
cost, not pressing the call: D1 is an architecture decision and it is yours.
What I would ask instead is narrower and answerable in a line — **if D1 is going
to stay open past Sunday, say so**, and the loop will stop sizing its week
around a quota it cannot spend.

---

*Method note: every number above is from a command run during this audit —
`experiments.coverage`, `run status`, `run verify`, an AST parse of both trees
at `ea2bdbf` and HEAD, direct reads of `ledger.json` / `gpu_budget.json` /
`/data/t301_shuffle_probe.json`, `ps` against the LC.03 workers, and
`/data/jack-logs/ladder.log`. Nothing is quoted from a commit message without
being checked against the artifact it describes.*
