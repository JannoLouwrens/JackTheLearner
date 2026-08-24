# OVERSIGHT — 26th audit, 2026-08-24 06:45 UTC

## VERDICT: ON TRACK — with one ledger finding and one structural cost that is now two-for-two

The ledger is sound. `run verify` re-judged all 83 auditable PASS entries from
the record alone and returned **zero** failures on all five probes; every PASS
has an implementation, a live commit, and a control that ran. Exactly **one**
gate moved in the loosening direction in seven days, and it was pre-registered
before the number, ordered by the previous audit, and compensated by a new
deterministic gate — that is the correct way to do it, and it still leaves the
sole `sight` claim resting on a control that measured dead (RANK 1).

The larger cost is not integrity, it is **the calendar**: the loop went fully
dark Fri–Sun for the second consecutive week, and for the second consecutive
week the free Kaggle quota expired on the Sunday inside that blackout. The two
clocks are anti-phased by construction. **30.9 free GPU-hours have now expired
unspent across W32 and W33** (RANK 2).

---

## 0. Is the ladder the right ladder? — coverage clean, 13th straight audit

`experiments.coverage` exits 0. **0 of 23 constitutional commitments has no
declared spec.** The 2026-08-10 miss stays closed. Registry grew 169 → 179
yesterday (NE.00–NE.09), the first growth in 11 days.

**9 of 23 commitments have a claim-kind PASS; 14 have none.** Unchanged from
the 25th audit — NE.00 passed, but it is declared `hunger/thirst (rule)`, so it
correctly buys no claim credit. Six of the fourteen show a passing spec the tool
refuses to credit because it is a fixture, rule or sensor (SM.01, VO.01, BA.01,
PS.02, PS.01/NE.00, XL.00) — the world can do the thing; nobody has shown *Jack*
does.

The two GOAL.md names as most likely to be quietly neglected are still the two
that are:

- **Curiosity — 1 of 11 claim specs passing** (T2.08). T5.06, *"unprompted
  exploration is real"* — the ladder-and-apple sentence itself — is NOT_RUN.
- **One brain / unison — 1 of 18 claim specs passing** (UB.9, whose own
  docstring records a hole in it; see B3). UB.7, the headline UNISON claim, is
  NOT_RUN.

**Why they are neglected is now measurable, and it is not the builder's
choosing** (see §3.2): 7 of the 11 curiosity claim specs and 9 of the 18 unison
claim specs sit behind **T2.01**, which sits behind **D1**, which is on the
owner's desk on day 15.

---

## RANK 1 — Sight's only claim rests on a control that measured dead. This was declared, not hidden.

**The finding.** `T3.01` (PASS, 2026-08-21T01:28:42, commit `f702251`) is the
sole claim-kind PASS behind the `sight` commitment. Its shuffled-label control
recorded, per seed:

    acc_shuffled_train  0.25 / 0.3167 / 0.25     reference line 0.35
    acc_shuffled        0.25 / 0.2233 / 0.25     gate |acc-0.25| <= 0.10  PASS

Two of three seeds sat at **exactly chance on their own shuffled training set**.
The arm never fit the data it was given, so its at-chance *test* reading proves
nothing about a distributional leak. The spec's own docstring says this in
plain words (`t3_01_ablate_vision.py:56-63`): *"with liveness un-gated, an
at-chance shuffled TEST reading is NOT evidence of no distributional leak."*

**The gate that would have caught it was removed 16 minutes before dispatch.**
`SHUFFLE_FIT_FLOOR = 0.35` was added as a VOID gate on 08-20 (`8f6f750`, this
desk's 23rd-audit B1), fired on its first real exercise (attempt 4 VOID,
`a9b94e9`), and was demoted to a recorded diagnostic on 08-21 01:12 (`f702251`).

**This is not silent loosening, and I will not call it that.** Every guard the
system owns fired correctly:
- The fork was **committed before any new number was seen**, per the 24th
  audit's explicit order (fate (i) vs fate (ii), both written out).
- Fate (i) was rejected **on measurement**, not on convenience: the liveness
  probe (`4b9ff30`, 0.70 GPU-h across three kernels) showed the best row needs
  >124 epochs — twice the claim budget — to clear 0.35, and seeds 0/2 reach no
  memorisation at *any* tested lr in the doubled budget.
- A **deterministic replacement** shipped in the same commit: sha256 over every
  raw train/test frame per seed; `hash_overlap_max` read 0.0.
- The constant was left in the file **unmoved**, the VOID it caused stays in
  history, and the blind spot is written into the docstring.

**And an actual leak is implausible, on evidence that did run:** `acc_ablated`
and `acc_pixshuf` both read **exactly 0.25 on all three seeds**. A per-class
colour or intensity artefact survives pixel-shuffling and would have shown
there. What remains uncovered is a spatially-structured per-class rendering
artefact — which on a shape-classification task is close to being the task.

**So why is this RANK 1?** Because "implausible on three side-arguments" is
precisely what the shuffled arm existed to replace with one measurement, and
because the registry's `control` field — the string an auditor greps, and the
one `run verify` reads — describes only the live TEST-accuracy half. Nothing a
future reader greps will tell them this control is half-dark. That is a
one-sentence fix (B1) and no re-run is owed.

---

## RANK 2 — Second week running, the loop went dark exactly when the GPU quota was expiring

**The blackout.** The ladder loop ran 12 iterations on Fri 08-21 and stopped at
12:07. It then logged **`STOPPED at 94% weekly usage`** every hour for **65
hours** — zero iterations on 08-22 and 08-23 — resuming at 08-24 05:07 when the
Claude week reset.

**The stop was legitimate this time.** I checked the meter the 25th audit
caught being misread. `scripts/claude_usage.py --pct` returns `week:all models`
(`claude_usage.py:64`), which is the gate `lib_usage.sh` reads. Live now:
session 17%, `week:Fable` 4%, `week:all models` **3%**, `--pct` = 3. The B1 fix
landed (`99b75b3`) and held. **No finding against the builder's meter reading.**

**The oversight went down with it, exactly as predicted.** The 25th audit warned
*"burning it to 90% takes the auditors down with you."* It did: **11
consecutive overseer audits refused at the gate** (08-21 12:37 → 08-24 00:37).
This report is the first in **71.7 hours** against a 6-hour cadence.

**The structural finding — one budget empties before the shared week does, and
this is now two-for-two:**

| | loop dark | Kaggle week ended | Kaggle charged | **expired unspent** |
|---|---|---|---|---|
| W32 | Fri 08-14 (partial) → Tue 08-19 | **Sun 08-16** | 21.185 h | **8.82 h** |
| W33 | Fri 08-21 12:07 → Mon 08-24 05:07 | **Sun 08-23** | 7.892 h | **22.11 h** |

Both budgets run Monday-to-Monday — the Claude pool resets Mon 05:00 UTC, and
Kaggle hours are accounted by ISO week (Mon–Sun). They are not anti-phased; the
problem is simpler and harder to fix. **The Claude pool empties on Thursday or
Friday and the Kaggle pool does not**, so the last two to three days of every
shared week hold free GPU quota with no agent awake to dispatch it, and the
quota then expires on the Sunday inside the blackout. **Total lost: 30.9 free
GPU-hours across two weeks**, neither time through misspending, both times
through not being awake to spend. (W32's outage ran on past Monday for a
separate, already-recorded reason — the D5 owner-resume expiry — but the Sunday
expiry landed inside it just the same.)

This is not a repeat of the 25th audit's meter bug. That was an error; this is
geometry. It needs an owner policy, not a builder fix — see FOR THE OWNER §2.

---

## RANK 3 — `COVERS:` markers in test docstrings are unvalidated, and three of them are wrong

`coverage.declarations()` reads `Spec.notes` **only** (`coverage.py:183`). The
parallel convention of writing `COVERS:` into the test file's docstring is
therefore read by nothing. Measured across all of `experiments/tests/`:

| spec | test docstring says | registry notes say | effect |
|---|---|---|---|
| **T2.04** (PASS) | `action head (claim)` | *no COVERS at all* | "action head" is not a commitment; buys nothing |
| **T2.05** (FAIL) | `world model (claim)` | `fast/slow (fixture)` | "world model" is not a commitment; disagrees |
| **T2.03** (PASS) | `sight (claim)` | `sight (fixture)` | credited as fixture; docstring overstates |

None of these inflates the coverage report — the registry is authoritative and
in every case the *conservative* side, which is why `check()` still exits 0 and
`declarations()` returns an empty `bad` list. **Damage to the ledger: none.**

But this is `coverage.py`'s own sentence — *"a marker that buys nothing while
looking like a claim is scar 2 wearing a new hat"* (`coverage.py:49-51`) —
occurring in the one location the module was never pointed at. A human reading
`t2_04_behaviour_cloning.py` sees a coverage declaration that no instrument
honours. Fix in B2. (The `shelterr` / `(fixure)` strings in
`t0_21_coverage_audit_honest.py` are deliberate fixtures for the guard spec and
are correct as written — I checked.)

---

## 1. Integrity of the ledger — CLEAN

84 PASS / 6 FAIL / 3 VOID over 179 registered specs.

    PASS rows with no implementation in experiments/tests/     0 / 84
    PASS rows whose `commit` is missing from git               0 / 84
    PASS rows absent from the registry                         0 / 84
    PASS rows with a declared control and no control_metrics    0 / 84
    PASS rows whose spec declares no control                   2  (T0.01, T0.10)

`run verify` independently re-judged 83 entries (T0.18 self-excludes, correctly)
and probed 81 controls: **0 verdicts that no longer re-derive, 0 gates that
ignore their control, 0 controls declared but never run, 0 gates that could not
be replayed, 0 entries that could not be audited.**

T0.01 and T0.10 remain the two existence claims with no control to delete —
structural, long-recorded, and not a new finding.

Two staleness alarms stand, both honest and both flagged by the system itself:
`BA.02` (VOID, test changed since the run) and `T2.02` (VOID, pre-`impl_sha`
content drift). Neither is a PASS.

---

## 2. Thresholds and controls over seven days — ONE loosening, loudly declared

I diffed `registry.py`, `registry_expansion.py` and `experiments/tests/` across
21 commits since 08-17 and read every changed numeric constant, gate and
`_check` branch.

**Loosened: exactly one** — T3.01's `SHUFFLE_FIT_FLOOR` VOID gate → recorded
diagnostic (RANK 1 above). Justified with a measurement, pre-registered as a
fork before the number, compensated by a new gate, constant left unmoved.

**Strengthened, same window:**
- `T2.05` null replaced with `min(persistence, mean)` per seed, plus a new ridge
  reference-arm gate — the docstring **predicted its own FAIL before the run**
  (`ecf92cc`), and it duly failed. Anti-run-until-pass, exemplary.
- `T3.01` gained a deterministic sha256 train/test disjointness VOID gate.
- `XL.01` `ALIEN_MIN_DIST` 1.5 → **2.0** (restored, tightening).
- `T2.07` and `T3.01` both had their `control` field *declared* in the registry
  after `protocol.py` refused an undeclared control at dispatch — the guard
  worked, twice.
- `UB.10` leak-gate instruments gated on their own liveness.
- `T0.24` and `T0.27` hypotheses widened, strengthen-only.

**Not found:** no seed count reduced anywhere (`seeds=3` throughout, no `seeds=`
line deleted); no `_check` gained an `or` in a claim branch; no assertion
removed; no control deleted. `LC.03`'s `_check` change was pure re-indentation
inside a loop.

---

## 3. Drift from the goal

### 3.1 What the builder did in the last 24 h — no drift

Two iterations, both `rc=0`:

| unit | GOAL.md sentence it serves |
|---|---|
| **LC.03 v2 harvest → VOID, screen CONCLUDED, D10 raised** | *"the exact balance as an empirical question the curiosity/needs bakeoffs decide, not a doctrine"* — and it refused to decide it |
| **NE.00–NE.09 registered; NE.00 PASS** | *"Jack has the needs of a human... the needs ARE the curriculum"* |
| **Field watch wk4** | *"Nature's solution enters as a bakeoff arm and must win on our substrate"* |
| **2 LESSONS entries** (ratchet-cap rule; cross-check runs both directions) | *"protects the honesty of watching what happens"* |

Nothing served none. NE.00 is a good unit: it proves the reward *algebra* Jack's
needs will train under is not farmable and is uniquely self-termination-safe —
a research-doc assertion converted into a ledger claim, at 4.7 s of CPU.

### 3.2 The converse — which parts of GOAL.md have no passing spec, and why

This is the harder question and the answer is now quantitative. Transitive block
mass, recomputed from the live registry:

    T2.01 = FAIL    blocks 36 specs
       -> 9 of 18  one brain / unison  claim specs (T4.01, UB.1-UB.8)
       -> 7 of 11  curiosity           claim specs (CU.1-CU.5, CU.7, T5.08)
       -> 2 of 2   plasticity          claim specs (T5.03, T5.04)
       -> 1 of 1   touch/contact, 1 of 1 tool use, 2 of 4 sleep, 1 of 4 generality

    LC.03 = VOID    blocks 8 specs
       -> 3 of 4   fast/slow           claim specs (DP.01, DP.02, DP.03)

    NE.01 = NOT_RUN blocks 8 specs (the whole NE family)
    UB.10 = NOT_RUN blocks 5 specs (3 more unison claim specs)

**65 of 179 specs are unreachable.** The PLASTIC-ONLY decree — the owner's own
2026-08-09 decree — has **both** of its claim specs behind T2.01. The two
commitments GOAL.md warns about are neglected because their specs are
mechanically unreachable, not because the builder chose easier wins.

### 3.3 A forward-looking risk worth naming before it costs a run

The NE family was registered yesterday into **W0** — the same world LC.03 v2 has
just measured as **unable to discriminate five learning cores at a 4× envelope**
(one arm at 3σ out of five, clean rig, ~190 core-hours). NE.03 (*"do needs teach
better than no needs"*, CPU_LONG) is a screening claim in that same world.
D10 option (b) — *"judge the world, not the cores"* — is therefore not only
LC.03's fork; it is a **precondition on NE.03 being informative**. NE.03's
registry notes carry the reward-exactness audit and the arms, but no line tying
its power to D10. Cheap to add now, expensive to discover after a CPU_LONG run.
Filed as B5, not as a finding against the registration — the cross-check that
was run (against XL.01/NE.08 and the §1.2 citation gate) was correct and caught
two real conflicts.

---

## 4. Is the builder alive and productive? — alive, and the stall was the gate, not the builder

    iterations, last 24 h                 2       (05:07, 06:07)
    ended rc=0                            2 / 2
    PASS delta, last 24 h              83 -> 84
    iterations, last 7 d                 55       (51 rc=0, 3 rc=1, 1 ABORT)
    iterations, 08-22 and 08-23           0       65 h of `STOPPED at 94%`

The single `ABORT: usage unreadable — refusing to run` (08-23 14:07) is the
correct behaviour: it refused rather than guessing. No repeated identical
failures, no paused loop nobody resumed (the week reset resumed it
automatically), no iteration aborting on load (load average 0.05, 80 GB free on
`/data`).

Both post-resume iterations were substantive rather than catch-up: one harvested
a 63-hour run and honoured its pre-registered fork, one grew the registry for
the first time in 11 days and landed a PASS. **No finding.**

---

## 5. Compute honesty — spending is honest; the waste is unspent hours, not misspent ones

    W33 (Aug 17-23)   22 jobs   7.892 h charged of 30    0.258 h failed (3.3%)
    W34 (current)      0 jobs   0.000 h charged of 30    fresh since Sun 08-23

W33's 7.892 h produced **8 Kaggle-backed ledger rows** (T2.06, T2.03, T2.04,
TA.02, T2.05, T3.01, T4.02, T2.07) plus four pre-registered diagnostic probes
(T3.01 curves, T3.01 shuffle-liveness, UB.10 pilot, UB.10 recipe) — each of
which had its decision rule written **before** launch and each of which changed
a design. `overruns` is empty. Failed-kernel hours are accounted separately and
their root causes are in the log (a mujoco sdist-before-wheels race, a hardcoded
`chdir` breaking a remote import) with LESSONS entries for both.

**There are no GPU hours without something to show for them.** The finding is
RANK 2: 22.11 h of W33 and 8.82 h of W32 were never dispatched at all.

---

## 6. Stuck decisions

**Open on the owner's desk, with age:**

| | raised | age | blocks |
|---|---|---|---|
| **D1** — does the 57M trunk stay in the control path | 08-09 | **15 d** | T2.01 → **36 specs** |
| **D7** — MovementMoodCoupling failed its ablation twice | 08-13 | 11 d (**ready to decide for 84 h**) | nothing; a component keeps unearned parameters |
| **D8** — BA.02 unmeasurable in the rover body | 08-14 | 10 d | balance's only claim spec |
| **D9** — the body fork (bakeoff table attached) | 08-21 | 3 d | folds into D1 |
| **D10** — LC.03 concluded with one learner | 08-24 | new | LC.04/05, OP.01, PS.04, DP.01-03 |

**Ready to decide with no further measurement possible: D7 and D9.** D7's re-run
on current code came back bit-identical; D9's bakeoff was run on free CPU and
the table is attached (arm C upright 100.0% on all three seeds vs the as-built
rover's 0.2–0.4%). Neither needs another number.

**Could the system have resolved anything itself with a bakeoff?** D9's bakeoff
*was* run — and correctly **not adopted** (`e9cc914`: *"table attached to D9,
NOTHING adopted"*; `9b2cf8f`: *"every winner needs owner adoption"*). That is
the right line and the builder held it under pressure.

**Was any owner decision quietly acted on?** I looked specifically for this and
found **no instance**. D9's winner was measured and left unseated. D10's
`wm-latent` was the sole 3σ learner and was explicitly **not** seated —
CHAMPIONS.md's learning-core seat still reads *"PPO — DEFAULT, never
defended"*. D8 blocks BA.02 and BA.02 remains VOID rather than being re-aimed at
a different body.

---

## 7. Bakeoff hygiene — clean, and D10 is the best-handled call on the page

`DECISIONS_RESOLVED.md` holds three entries; I re-read all three. PS.01/J is
recorded as a VOID and is **not** treated as a verdict; PS.01/J2 names a winner
(`impact_speed`) with its learning gate cited; D2 was resolved by ledger replay
with the reasoning attached. No winner chosen inside a noise margin.

**The live case is LC.03, and it is exemplary.** A screen with no re-screen cap
is a ratchet; the 25th audit ordered the fork be committed before the number,
and it was — 2.5 days before the run landed. When the number came back with one
learner, fate (ii) fired: **no v3, no envelope growth, no re-roll**, and the
`data_starved` flags on three arms were priced *in advance* (the 3σ requirement
grows with added lives as fast as the projected gain does). The finding was
recorded as a result about the world, and the design fork was routed to the
owner rather than resolved at the desk.

One hygiene nit, not an integrity issue: **CHAMPIONS.md's learning-core cell is
now stale** — it still reads *"a 4× envelope re-screen (v2) was launched 04:24
and is IN FLIGHT, pid 310395 + 3 workers, ETA ~Aug 23 late."* It concluded
2026-08-23T21:11. The seat itself is correctly unfilled. Filed as B4.

---

## 8. The honest summary — are we closer to a curious humanoid, or to a longer list of green ticks?

**Closer, but by a smaller step than the +1 suggests, and for a reason worth
saying plainly.**

Yesterday's single PASS was NE.00 — a 35-state tabular MDP proving that
drive-reduction reward is the unique form under which self-termination is never
optimal and closed cycles cannot be farmed. That is a *good* tick: it is the
kind that forecloses a whole class of future silent failure (NetHackEat ships
the exploit NE.00 makes impossible here), and it cost 4.7 seconds. But it is
apparatus for a curriculum, not a creature doing something.

The step that actually mattered in this window was a **VOID**. LC.03 spent 190
core-hours and ~63 wall-hours to learn that W0 — the world Jack currently lives
in — cannot tell five learning cores apart, and that exactly one of them
(`wm-latent`) learns to survive at 3σ while four do not. Then it refused to buy
a second learner with more compute, because it had written down beforehand what
it would do if it only got one. **A system that will spend three days to earn a
VOID and then honour it is the thing this project is actually trying to build.**
The green ticks are downstream of that discipline; they are not a substitute for
it.

Against that: the ladder-and-apple standard has not moved. **T5.06 —
"unprompted exploration is real" — has never run.** UB.7, the headline unison
claim, has never run. Both of GOAL.md's plasticity claim specs have never run.
Not because anyone deprioritised them: 65 of 179 specs are mechanically
unreachable, and the largest single cause is a fifteen-day-old decision about
whether a 57M trunk stays in the control path. The builder cannot climb past
this and has correctly stopped trying to.

So: the ladder is honest, the instruments work, the builder held every line it
was asked to hold under a three-day blackout. What is missing is not rigour and
not effort. **Jack still has no passing spec that shows him wanting anything.**
The one arm that learned to survive learned it in a world we have just measured
as too shallow to distinguish learners — and the two decisions that would let
that finding matter are both sitting on a desk, not in a queue.

---

## FOR THE BUILDER

Ranked. B1 and B2 together are under an hour and neither owes a re-run.

**B1 — Record T3.01's control blind spot where an auditor greps, not only in
the docstring.** The registry's `control` field for T3.01 accurately describes
the live TEST-accuracy gate and says nothing false — but it also says nothing
about the half that was demoted, and `Spec.control` is what `run verify` and
every future reader read. Append one sentence, e.g.:

> *"`acc_shuffled_train` is RECORDED, not gated (v3, 2026-08-21): the registered
> run read 0.25 / 0.3167 / 0.25 against the 0.35 reference line, so this control
> is NOT evidence against a distributional leak. Identity leakage is carried by
> `hash_overlap` (0.0) and the pixel-shuffled arm (0.25 every seed)."*

Doc-only; use the amendment lane you built at `69f8f69`. **No re-run owed, and
do not re-run to get a better number** — the measurement is correct as recorded.

**B2 — Validate `COVERS:` in test docstrings, or stop writing them there.**
`coverage.declarations()` reads `Spec.notes` only (`coverage.py:183`), so the
docstring convention is honoured by nothing. Three are wrong today (RANK 3):
`t2_04` declares `action head (claim)` — not a commitment, and its registry
notes carry no `COVERS` at all; `t2_05` declares `world model (claim)` against a
registry `fast/slow (fixture)`; `t2_03` declares `sight (claim)` against a
registry `sight (fixture)`. Cheapest correct fix: **extend T0.21** with a
property that every `COVERS:` marker found in an `experiments/tests/*.py`
docstring (a) resolves to a name in `COMMITMENTS` with a valid kind, and (b)
matches its spec's registry declaration exactly. That is the guard spec's
existing job, one directory over. Then fix the three.

**B3 — carry-forward from the 25th audit's B4, still prose.** UB.9's registry
notes read only `COVERS: hearing (claim), one brain / unison (claim)`. It is the
**only** claim-kind PASS behind unison's 18 claim specs, and its own docstring
(lines 101-102) records that a per-arm recipe pathology — UB.10's *measured*
disease — is not excluded by its design. Either a same-run must-learn target per
unimodal arm, a recorded per-arm loss descent, or an explicit registry statement
that UB.9's claim is conditional on the shared-trainer argument. It still does
not need to be done this week. It still needs to stop being prose.

**B4 — CHAMPIONS.md's learning-core cell is stale by three days.** It reads
*"v2 ... IN FLIGHT, pid 310395 ... ETA ~Aug 23 late"*; LC.03 concluded
2026-08-23T21:11 with `void_reason: "fewer than two learners (1 cleared)"`.
INTEGRATION_QUEUE protocol step 6 makes updating the seat the harvester's duty.
**Do not seat `wm-latent`** — that is D10 option (a) and it is the owner's. Just
make the cell say what happened and that the seat is pending D10.

**B5 — tie NE.03's power to D10 before it is implemented, not after.** NE.03 is
a CPU_LONG screening claim in W0, and LC.03 v2 has just measured W0 as unable to
separate five cores at a 4× envelope with a clean rig. If D10 resolves toward
(b) — redesign W0's discriminability — NE.03 as registered is measuring the same
shallow world. One line in NE.03's notes, in the same idiom you used to bind
NE.08 to XL.01's power verdict: *a pre-run power calculation against LC.03 v2's
recorded spreads is required before dispatch.* No threshold moves.

**B6 — dispatch-then-idle, and mean it this time.** The meter fix landed and
held (verified: `--pct` returns `week:all models`, 3% now). But between the
correction at 07:17 on 08-21 and the blackout at 12:07, only **0.29 GPU-h** was
dispatched — and 22.11 h then expired on the Sunday. Kaggle kernels and
`launch_detached.sh` compute *through* a blackout and write their own receipts.
When `week:all models` passes ~80%, the correct iteration is a lean one that
**dispatches everything W-week-worth and then idles**, not a full one that plans
work for after the reset. W34 is fresh at 30 h and the week is young; the
zero-pass ranking from `run coverage` is where those hours belong.

---

## FOR THE OWNER

Three items. Only §2 is new, and it is a policy question, not a science one.

**1. D1 is fifteen days old, and I can now name exactly what it costs — not in
specs, but in your own words.**

T2.01 is FAIL and blocks **36 specs** transitively. Sorted against the
commitments *you* wrote into GOAL.md:

- **both** claim specs behind the PLASTIC-ONLY decree (T5.03, T5.04) — your own
  2026-08-09 decree, with zero evidence behind it and no route to any;
- **9 of the 18** claim specs behind *"one brain, all senses in unison"* —
  including UB.7, the headline unison claim;
- **7 of the 11** claim specs behind curiosity — including CU.1–CU.7, the family
  that would show him exploring because he wants to;
- the only claim spec for touch, and the only one for tool use.

The ladder-and-apple sentence — *"he must try to climb the ladder, fall, and
learn from falling, purely out of curiosity"* — has no passing spec, and the
specs that would give it one are behind this decision. I am not re-arguing the
options; the evidence table in D1 is complete and unchanged. **If D1 is going to
stay open, saying so is itself useful** — the loop will stop sizing weeks around
work it cannot reach.

**D7 and D9 are both ready and neither needs another number.** D7 has been
ready for 84 hours and takes ten seconds: delete MovementMoodCoupling, redesign
it, or **accept it on the record as cosmetics** — (c) is a legitimate answer and
closes it. D9 has its bakeoff table attached (arm C: upright 100.0% of decisions
on all three seeds, hand reach 1.165–1.185 m of a ~1.19 m ceiling; the as-built
rover: 0.2–0.4%), run on free CPU and deliberately **not** adopted.

**2. NEW — one of your two budgets empties three days before the other, and it
has cost 30.9 free GPU-hours in two weeks.**

Your 90% weekly stop is working exactly as written and I am not asking you to
weaken it. The problem is that the two budgets run on the same Monday-to-Monday
week but drain at different rates:

- the **Claude** pool is exhausted by Thursday or Friday;
- **Kaggle's** 30 free GPU-hours are not, and they expire the following Sunday.

So the last two to three days of every week hold free GPU quota with nobody
awake to dispatch it, and the quota dies inside the blackout. Twice in a row:
**W32 8.82 h expired, W33 22.11 h expired.** Neither loss was misspending —
every hour actually charged produced a ledger row or a pre-registered diagnostic
that changed a design. The loss is entirely "nobody was awake to press the
button."

This is arithmetic, not a bug, so it will recur every week until a policy
changes it. The cheapest fix costs you one sentence and no relaxation of the
stop:
**authorise a standing "dispatch-then-idle" carve-out** — when `week:all models`
crosses ~80%, the loop may spend one lean iteration dispatching detached Kaggle
work before it freezes. Kernels compute through a blackout for free and write
their own receipts. I have filed the builder half as B6; the carve-out is yours
because it is a spend of your Claude budget, however small.

There is a second-order cost you should know about: the same stop takes the
**auditors** down with the builder. This report is the first in 71.7 hours
against a 6-hour cadence — 11 consecutive overseer audits refused at the gate,
and the same for the Review. The 25th audit predicted this in writing three days
ago. Whatever you decide about the carve-out, the oversight organs going dark
for three days at a time is worth deciding *deliberately* rather than inheriting.

**3. D10 is new on your desk and it is a real fork, not a formality.** LC.03
concluded with exactly one learner (`wm-latent`, t=4.65 vs null, +92 s
cross-life transfer) out of five, on a clean rig, after 190 core-hours. The
builder correctly refused to seat it, refused a third re-screen, and wrote down
why before it saw the number. The three options are in `DECISIONS_NEEDED.md`.
My only addition: option (b) — *judge the world, not the cores* — is worth more
than it looks, because the **entire NE family was registered into that same
world yesterday**. If W0 cannot separate five learning cores, it is a live
question whether it can separate "with needs" from "without needs" either. That
makes (b) a decision about NE.03 as well as about LC.04.

---

*Instruments run this audit: `experiments.coverage` (exit 0),
`experiments.run status`, `experiments.run verify` (83 entries, 81 controls, 0
failures), `experiments.run blocked`, `git log -p --since="7 days ago"` over
`registry.py` / `registry_expansion.py` / `tests/` (21 commits), a per-PASS
implementation/commit/control cross-check, a docstring-vs-registry `COVERS:`
diff over all of `experiments/tests/`, a transitive block-mass computation
against `coverage.declarations()`, `gpu_budget.json` per-week reconciliation
against the ledger, and `/data/jack-logs/{ladder,overseer,review,field_watch}.log`
cadence counts.*
