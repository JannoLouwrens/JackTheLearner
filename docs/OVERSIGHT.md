# OVERSIGHT — 62nd audit, 2026-09-02 12:44 UTC (HEAD `315ede4`, 0 unpushed, tree CLEAN)

## VERDICT: DRIFTING — `T0.27`'s live violation count moved **2 → 3** at 09:18 today and nobody said so, three days before `D16` fires on the two-violation premise; and **two of the three violations are recoverable** while the instrument reports all three with the sentence *"that implementation was never committed"*

No integrity breach. The ledger is the most trustworthy it has been: 94/94 PASS
rows are mechanically sound, and I could not find a single loosened threshold in
seven days of diffs. The drift is elsewhere — the ladder gained **one** spec in
24 hours across **12** iterations while its number-one blocker went untouched
for a **24th** day, ~10.8 free GPU-hours expire Sunday for the **fourth**
consecutive week, and the guard that exists specifically to catch
threshold-moving is quietly accumulating violations faster than they age out.

Findings are ranked by damage to the trustworthiness of the ledger.

---

## THE FOUR MANDATORY INSTRUMENTS

| instrument | rc | reading |
|---|---|---|
| `coverage` | **2** | **0** commitments with NO spec. 4 CLAIM-DEAD (smell, balance, shelter/building, thermal); 8 more with live claims and nothing passing. 4 NEW unrunnable GOAL.md citations (GEN.02/03/06/09, all welded behind LC.07). Standing red by design. |
| `decisions --check` | 0 | **0 MEANS-ESCALATED, 0 UNDECLARED, 0 OVERDUE.** D15/D16 due 09-05, D17 due 09-07. Ratchet 0/10. |
| `champions --check` | 0 | 27 seats, **8 declared violations**, `arena_missing 0`, unfalsifiable 3/3, unverified verdicts 2/2. Every ratchet at or below baseline. |
| `run review-queue` | 0 | **0 violations.** 26 OPEN / 2 HELD / 2 ACTED of 30 routed; oldest live 9 d; consumer ran today. |

**All four ratchets are green and I verified each reading rather than taking the
journal's word for it.** `decisions --check` printing only D15–D17 is correct,
not a blind spot: the eleven defaults armed for 2026-08-31 all fired on 09-01,
each with the SYSTEM.md-required journal line, each naming a concrete artifact I
spot-checked (D1 → `D1.0` registered and run; D10 → `LC.07` registered; D8 →
`BA.02` re-parented behind `LT.08`). The 41st audit's warning that *"nine of the
eleven cannot be fired by the organ instructed to fire them"* was heeded — the
**builder** fired them, not the overseer.

---

## FINDING 1 (RANK 1) — the anti-threshold-moving guard gained a third violation today, unreported, and the instrument mis-describes two of the three

`T0.27` — *"A threshold moved after a FAIL leaves an artifact, not a paragraph"*
— is the spec that stands between this project and its worst possible failure.
It is RED by the owner's pending decision `D16`, which **fires in three days**
on option (b): *"the warning stands, `T0.27` stays RED and is not touched, and
the red is reported in every status until **the pair** ages out of history."*
Singular. Both `D16` and my own 60th-audit evidence update say **two**.

**Live now, read from `protocol.audit_supersedes_fail` against the real
ledger — 3 violations, 8 checked pairs, 24 unauditable:**

| spec | status | stamp | recorded |
|---|---|---|---|
| `T0.17` | FAIL | `d84101e+dirty` | 2026-08-29T13:14:23 |
| `LG.00` | VOID | `8faff43+dirty` | 2026-08-30T18:47:59 |
| **`T0.29`** | **FAIL** | **`661a48f+dirty`** | **2026-09-02T09:18:06** |

The count's history, from `T0.27`'s own ledger rows: **1** (08-29, `4e8577d`) →
**2** (08-30, `f4115f2`, held for ten consecutive runs) → **3** (today,
`5c8d18b`). Checked pairs 5 → 7 → 8.

**Two things went unreported, and they are different failures.**

**(a) The number moved and the commit said it hadn't.** `T0.29`'s dirty FAIL was
created at 09:18 by the 61st audit's own B4 work (`661a48f`, the `champions.py`
VERDICT conjunct). The T0.27 re-buy two hours later recorded `live_violations:
3`; its commit `965f54a` describes it as *"the deliberate FAIL"* and the journal
as *"honestly re-recording its deliberate D16 FAIL"* — as though the row were
unchanged. It was not. This is not dishonesty; it is the row being read as a
known-red token rather than as a number. **A gate that is deliberately RED stops
being read as a measurement** — which is precisely the newest lesson in
`LESSONS.md` (*"a violation that buys a RED is invisible to all of them"*)
recurring two days after it was written.

**(b) The 12:07 slot repaired the wrong instrument and reported success.** It
re-ran `T0.29` from a clean tree, correctly, and reported *"dirty-stamp block
now empty."* That is true of `run status`'s dirty-stamp check. It is **not** true
of `T0.27`: the +dirty FAIL row sits in `history` and no re-run removes it —
exactly as `D16` already explains for `T0.17`. The incident was repaired in one
instrument and remains permanent in the other, and only the first was reported.

**And the instrument's own sentence is now false for two of three.** All three
violations are reported as *"that implementation was never committed."* The
failing bytes are preserved for two of them, and the ref existing is proof
rather than assertion (`preserve_impl_bytes` refuses to write unless the
re-derived sha matches the row):

    refs/jack/failimpl/LG.00/2026-08-30T18-47-59  -> blob d39a0ef
    refs/jack/failimpl/T0.29/2026-09-02T09-18-06  -> blob facfff9

Only `T0.17` is genuinely unrecoverable (checked with `tree_reconstructing_sha`
on 08-30). So the `git diff` the rule demands is **possible** for two of three
violations, and the instrument cannot say so.

**Why this ranks first.** `D16` fires 09-05 and `t027-preserved-failimpl-as-artifact`
is DUE 09-05 — well sequenced by the builder. But both currently state a
two-violation, one-recoverable premise that is 24 hours out of date, and the
rate now has three points across three distinct specs (08-29, 08-30, 09-02).
`D16`'s own forecast was *"this fires more, not less"*; option (a) assumed the
pair would *"age out ... soon-ish."* At ~1 new violation per 1.5 days against a
20-entry history, **they are arriving faster than they age out.** The owner
should rule on today's number, not on the 08-29 snapshot.

*Nothing here asks for the gate to be relaxed, and I have touched no gate, no
default and no `decide_by`.*

---

## FINDING 2 (RANK 2) — the 38-spec blocker has not been touched in 24 days

`run blocked` is unambiguous and has been for weeks:

    T2.01 = FAIL   frees 35  (blocks 38)  — Locomotion beats a random policy
    NE.01 = FAIL   frees  8  (blocks  8)
    LT.01 = FAIL   frees  7  (blocks  9)  — The Ladder Test

`T2.01` blocks more than the next four blockers combined. Behind it sit **the
entire curiosity family** (CU.1–CU.7), **the entire unison family** (UB.1–UB.8),
and **seven of the Tier-5 claims** — that is, GOAL.md's thesis. Its
implementation was last modified **2026-08-09** (`49529e6`); its last run was
5.6 h across 3 seeds, FAIL, and its own registry note says the first healthy run
*"failed only effect size, 2.21 sigma at 192K steps/seed"* with the curve still
climbing at cutoff.

`LT.01` — *the Ladder Test*, GOAL.md's own image of success, the ladder and the
apple — is also FAIL, blocking 9.

**This is now owned, and only since yesterday.** The 60th audit found that
`D1.0` burned 16.17 GPU-hours (54% of a weekly quota) to a VOID and that *"no
row, no `DUE:` and no priority line owned fixing the arm"* while `T2.01` waited
on its winner. The builder routed `d10-successor-rerun-under-adopted-gate`
(DUE 09-08) in response, and the queue row's arithmetic that the VOID is **not**
foreclosed is sound: `c_e2e` returned 404.3 vs random's 108.7 (a 3.7× gain)
scored against its own wider spread, with untrained twins at 2.94–2.96σ against
a 3.0 bar. That is a gate-scoring artifact, not an envelope wall.

**The finding is the latency, not the ownership.** The chain is: gate design
adopted 09-06 → consequence stamped 09-08 → `D1.0` attempt 2 dispatches in W36 →
`T2.01` re-runs after that. Thirty-eight specs — including every falsifiable
claim behind curiosity and unison — stay blocked for at least four more days on
top of twenty-four. No instrument on this board ranks by *how long the top
blocker has been the top blocker*, so the 24-day figure appears nowhere.

---

## FINDING 3 (RANK 3) — a constitutional sense was parked behind a node that is unreachable by its own registry's arithmetic

`coverage` correctly reports **balance** as CLAIM-DEAD. What no instrument says
is that its designated revival path cannot be walked.

`D8`'s armed default fired 2026-09-01 and parked `BA.02` *"until a body with
directional catch authority exists"*, re-parenting it behind `LT.08`. `D9` fired
the same day and parked the body question *"until the playground-humanoid line"*
— which is the same `LT.08`. Then:

    LT.08  depends_on = [LT.07, T2.01, T2.02]      T2.01 = FAIL, T2.02 = VOID
    LT.07  <- LT.04 <- LT.03 <- LT.01 = FAIL

and `LT.08`'s own registry note prices it:

> *"at T2.01's measured ~128 env-steps/s a 20M-step arm-seed costs 43 h, so
> 3 seeds exceed a whole week of Kaggle quota for ONE arm. The prerequisite is
> a throughput spec, not more quota."*

So a constitutional sense is parked behind a node that is (i) blocked by the
two specs the park was meant to route around, (ii) additionally blocked by the
FAIL of the Ladder Test, and (iii) priced by its own note at more than a week of
free quota for a single arm — against a hard `SYSTEM.md` free-compute-only
constraint.

**I want to be fair about what this is and is not.** Parking is an
already-permitted action, so the defaults were legal; both were correctly
armed, correctly journalled, and each is individually defensible on its
evidence. `BA.03` was registered as the live in-body successor and ran honestly
to a VOID. Nothing here was concealed. **The defect is compositional**: two
defaults fired the same day, both pointing at one node, and no instrument asks
whether a park's stated release condition is itself reachable. `run blocked`
models reachability; `coverage` models claim-liveness; neither joins them to a
park's release condition. The result is a commitment whose revival is
foreclosed by arithmetic while every gate reads green.

`five-commitments-are-claim-dead-behind-foreclosures` (DUE 09-11) is the right
desk for this, but its framing is foreclosures, not unreachable release
conditions.

---

## FINDING 4 (RANK 4) — a fourth consecutive week of expiring free GPU quota, and W35's spend bought no ledger movement

`gpu_budget.json`, charged hours by week against the 30 h/week Kaggle reset:

| week | charged | lost |
|---|---|---|
| 2026-W33 | 7.89 | ~22 |
| 2026-W34 | 1.62 | ~28 |
| 2026-W35 | **19.20** | **~10.8 projected** (expires Sunday 09-06) |

W35 is the best week since W31 — that is real. But of its 19.2 hours, **~17.9
bought no ledger advancement**: `D1.0` 16.17 h → VOID, `UB.10` ~1.25 h → VOID,
`LC.07` pilot 0.44 h → PILOT-BLOCKED. The remainder re-bought `T1.09`/`T1.10`
PASSes that were already PASS.

**The builder is right not to manufacture a dispatch**, and I want that on the
record: every GPU cost class is VOID-arm or pilot-blocked, `coverage` says so
explicitly, and inventing a run to spend quota would be exactly the behaviour
this system exists to prevent. It refused, and said why, in four separate
iterations. That is correct conduct.

The finding is the standing consequence: **free compute is being lost at ~20
h/week to a design backlog**, and 09-06 — the day the gate decision lands — is
also the day W35's quota resets, so the remaining 10.8 hours cannot be spent on
the thing they would unblock even in principle.

---

## FINDING 5 (RANK 5, section 7) — a VOID was seated as a verdict; it was foreseen, it fired unopposed, and the instrument gap is now closed

Section 7 asks directly whether any decision was made on a VOID. **Yes, one.**

`D10`'s armed default fired 2026-09-01 and seated `wm-latent` **BY VERDICT** off
`LC.03`, whose ledger status is VOID. `SYSTEM.md`: *"VOID: an arm failed the
learning gate; fix the arm, do not decide"* and *"two non-learners cannot
arbitrate an architecture."* The 41st audit wrote the prediction down in
`DECISIONS_NEEDED.md:3100` before it happened, including the second half:
*"afterwards `champions --check` prints Learning core BY VERDICT ok, because it
reads the table and cannot ask whether a verdict was earned."*

Both halves occurred exactly as forecast. The seat printed `ok` for five audits.

**The system has since repaired itself, and this is the healthiest thing in the
audit.** The 61st audit's B4 (executed today, `661a48f` → `81e3b97`) added the
`VERDICT-IS-A-VOID` and `VERDICT-UNDECLARED` classes, built class-before-migration
with the class firing 3/3 live first. `champions --check` now prints:

    [VERDICT-IS-A-VOID] Learning core — held BY VERDICT off LC.03 (VOID)

and the repair is clocked on the `d10-*` rows. The same work surfaced a second,
independent find the builder held open rather than hiding: **the World seat holds
`BY VERDICT` — the file's strongest marking — off the 4–6× Craftax comparison,
which has no spec id and no ledger row at all.** `CHAMPIONS.md:290` declares that
debt in the open rather than papering it with a false `VERDICT:` line. That is
law 1 applied to the project's own governance document, and it is the right call.

**What is worth recording is the mechanism, not the outcome.** A violation
foreseen in writing by one organ fired unopposed because the only organ watching
(the overseer) may not write the file, and the arming audit could tighten a
default but not rewrite it. The prediction was correct, on the record, and
structurally unable to stop anything.

---

## SECTIONS WITH NO FINDINGS — stated plainly, because they are true

**Section 1 — ledger integrity: CLEAN.** All **94 PASS rows** checked
mechanically: every `commit` still resolves in git (0 failures), every spec
declares a `control`, and every declared control has populated
`control_metrics`. The only two PASS rows with empty `control_metrics` —
`T0.01` (repo imports) and `T0.10` (Kaggle round-trip) — carry
`control="NONE, BY DECISION (52nd audit B5)"` on their face with the reasoning
attached. `T0.18` gates the re-derivability property. **Zero findings.**

**Section 2 — thresholds and controls: CLEAN, and better than clean.** Seven
days of `git log -p` over `registry.py`, `registry_expansion.py` and
`experiments/tests/`. Every constant that moved, moved in the tightening
direction:

- `N_LIVES 16 → 32` (T3.09 v2) — sample size, drawn sequentially so the old 16
  spawns are an exact prefix; every gate constant untouched; committed before
  the run.
- `TEMP 0.25 → 1.0` (LG.10 v2) — the parameter-free softmax default, chosen
  without previewing the T=1.0 draws; more entropy makes every gate strictly
  harder.
- `seeds 1 → 3` on T3.09 and the ME family — strengthening.
- Budget label `CPU_LONG → CPU_DAYS` (BA.03) — a timeout label, not a gate.

**No threshold moved in the loosening direction, no control was deleted or
weakened, no `_check` gained an `or`, no seed count was reduced, no assertion
was removed.**

I scrutinised the one edit that had the right shape to be gate-fitting — a
`_check` modified *after* a recorded run (`19461c4`, T3.09). It moves the
`shuf_gain` control-vacuity lane **above** the claim branch, so the lane fires
whichever way the claim went. Verified against the recorded numbers: attempt 3's
metrics now return **VOID** where they returned FAIL. **The repair costs the
ladder a demonstrated point and the commit message says so in capitals.** That
is the opposite of gate-fitting, and the builder did it against its own work.

**Section 4 — the builder is alive and honest.** 12 iterations in the last 24 h,
**12 of 12 ended `rc=0`**, no paused loop, no credit exhaustion (usage gate read
8% → 16% `week:all models` across the day, correctly reading the all-models line
per the standing rule). One `LEFTOVER=1` at 05:39 — the third in three days —
and the builder's response was the right one: it fixed the **cause**, putting
self-declaration into `run_spec` at the single choke point every spec run passes
through (`5c8d18b`, B3), rather than into a launch script a session could forget.
Round-trip tested against the real shell reader, 29/29, red-verified first.

**Today's honesty record is the strongest evidence in this audit.** The builder
bought **three REDs and one PASS**: `LG.10` FAIL (two attempts, both temperature
endpoints paid, no re-roll), `T3.09` FAIL→VOID (and refused to execute a 559-line
`kills` deletion on a run its own control impeached), `ME.11` SETTLED FAIL (the
honest RED the Review priced, bars untouched). It also declined to execute a
deletion clause it was entitled to execute, and it caught and killed its own
dirty-tree launch at ~1 minute.

**Section 6 — stuck decisions: nothing blocked that could be self-resolved.**
Zero MEANS-ESCALATED. Every open decision (D15, D16, D17) is armed with a
default and a date, and all three are goal-class — pace policy, an honesty-gate
disposition, and the PLASTIC-ONLY decree's own re-open trigger. None is a fork a
measurement could settle. **The D1 disease is not present.**

**Section 3 — no drift in what was worked on.** Every unit today traces to a
GOAL.md sentence: `LG.02` → *"his diary records whose advice proved true, so
trust in a person can be earned and checked"*; `LG.10` → *"the LLM is his mouth,
never his mind"*; `ME.11` → *"memory makes it him"*; `T3.09` → *"components must
EARN their parameters or be deleted"*; B1–B4 → *"the system is the product."*
Nothing served no sentence.

---

## SECTION 8 — THE HONEST SUMMARY

**Closer to a creature, by exactly one step — and it was a good one.**

`LG.02` is a genuine advance and it is not a green tick. Given two advisors whose
claims his own foraging verifies, Jack's advice-following diverged by track
record: **82% follow for the truthful voice, 13% for the liar** (divergence
0.689 ± 0.103 against a pre-registered 0.40 worst-seed gate, 3 seeds). The three
things that make it a trust claim rather than a rigged demo all held —
stripping attribution collapses divergence to 0.028 while the verification
machinery stays provably alive; the owner's swap control shows trust **migrating**
to the newly-truthful voice mid-life (0.711); first-encounter trust was exactly
the 0.5 prior for both voices on every seed. **The creature learned who lies to
him, from living with them.** That is the ladder-and-apple standard, in the
social domain.

**And the rest of the picture is the reason this reads DRIFTING.**

The board's green is concentrated where it is cheapest. Against GOAL.md's
constitutional core:

| commitment | specs | passing claims |
|---|---|---|
| **one brain / unison** | 23 | **1** |
| **curiosity** | 12 | **2** |
| fast/slow | 8 | **0** |
| sleep | 5 | **0** |
| plasticity | 4 | **0** |
| **shelter/building** — *the owner's own image of success* | 2 | **0, CLAIM-DEAD** |
| **thermal — *"too cold kills him"*** | 4 | **0, CLAIM-DEAD** |
| smell, balance | 4 | **0, CLAIM-DEAD** |

Twelve of twenty-three commitments have zero passing claim spec. The three
capabilities GOAL.md warns are *"most likely to be quietly neglected in favour of
easy wins"* — curiosity, all-senses fusion, learning-by-living — are 1/23, 2/12,
and blocked behind `T2.01` respectively.

**What today actually consisted of:** 12 iterations, 1 new PASS, 2 honest REDs,
and roughly seven units of machine maintenance — certificate re-buys (`T0.17`,
`T0.21`, `T0.27`, `T0.29` **twice**, `T0.31`), the `champions.py` conjunct, the
procwatch hook, the queue stagger. `SYSTEM.md` says plainly that a session which
makes the machine better at catching its own errors has done the whole job, and
by that standard the day was excellent — B1–B4 all closed, one real class of bug
made impossible. **I am not calling maintenance drift.** The finding is that
there was nothing else available: `coverage` reports **0 fresh dispatches** at
every cost class, and the journal's own handoff for the next iteration is *"a
`--gate` sweep or the `cpu<1min` class."*

**So the honest answer to "closer to a curious humanoid, or only to a longer list
of green ticks?" is: neither, today.** The list of green ticks grew by one, and
it was earned. But the machine is now polishing itself because the ladder has
nothing to hand it, and the reason the ladder has nothing is that **every road
runs through `T2.01`, `LC.03` and `W0`** — a body that cannot beat random, a
learning-core screen that found one learner, and a world whose own instruments
keep reporting that it is too shallow to measure what we ask of it. Six
independent specs now say this in their own recorded words: `DP.04` (*"no
resolution in W0"*), `SH.02` (*"D10 evidence that W0 is the bottleneck"*),
`UB.14` (*"the binding fault is the VENUE, measured"*), `BA.03` (*"the blind twin
holds 98.9% of the horizon"*), `T3.09` (*"the site rewards any perturbation"*),
`LC.03` (*"a REDESIGN of the screen or of W0"*).

**That convergence is the most important number in this audit and it has no
instrument.** Nine separate queue rows carry nine separate repairs for what may
be one cause. The 09-06 docket bundles four of them, which is the right instinct.
The system has correctly diagnosed itself one spec at a time; what it has not yet
done is add up the diagnoses.

---

## FOR THE BUILDER

**B1 — RANK 1, before 2026-09-05. Update `D16`'s evidence with today's number
and today's recoverability, then re-point the `t027` row at it.** The owner's
default fires 09-05 on a premise that is 24 hours stale.

- Append a dated evidence update to `D16` in `docs/DECISIONS_NEEDED.md`
  recording: **3 violations, 8 checked pairs, 24 unauditable**; the third is
  `T0.29 FAIL 661a48f+dirty 2026-09-02T09:18:06`, created by the 61st audit's
  own B4 work; and the progression **1 (08-29) → 2 (08-30) → 3 (09-02)** across
  three distinct specs.
- State plainly that **two of three are recoverable** —
  `refs/jack/failimpl/LG.00/2026-08-30T18-47-59` (blob `d39a0ef`) and
  `refs/jack/failimpl/T0.29/2026-09-02T09-18-06` (blob `facfff9`) — so
  `audit_supersedes_fail`'s single sentence *"that implementation was never
  committed"* is now false for the majority of the rows it prints. Only `T0.17`
  is genuinely lost.
- Note the rate against option (a)'s assumption: at ~1 new violation per 1.5
  days against a 20-entry history, **they arrive faster than they age out.**
- Add the same three facts to the `t027-preserved-failimpl-as-artifact` row
  (already DUE 09-05, correctly sequenced) so the desk and the default see the
  same number on the same day.
- **Change no gate, no default, no `decide_by`, and do not relax
  `audit_supersedes_fail`.** The question of whether a verified preserved
  manifest earns a second lane is the Review's, already routed, and the party
  proposing it is the party it would exonerate.

**B2 — the reporting defect that let B1 happen, and it is the generalisable
half.** Two separate reports today described `T0.27` as unchanged while its
`live_violations` metric had moved (`965f54a`; the 11:07 journal entry). And the
12:07 slot reported *"dirty-stamp block now empty"* — true of `run status`,
false of `T0.27`, where the same incident is permanent. Make one of these
mechanical rather than remembered:

- (a) have `run status` print `T0.27`'s `live_violations` **as a number with its
  delta since the previous row**, so a deliberately-RED gate still reports a
  moving measurement; **or**
- (b) have the dirty-stamp block in `run status` state, when it clears, that
  clearing it does **not** clear the `T0.27` pair the incident created, and name
  the pair.

(a) is the stronger repair: it generalises to every gate held RED by a pending
decision, which is the class `LESSONS.md`'s newest entry already names.

**B3 — join reachability to park release-conditions (FINDING 3).** No instrument
asks whether a PARKED spec's stated revival condition is itself reachable.
`BA.02` is parked behind `LT.08`, which is blocked by `T2.01`/`T2.02`/the LT
chain **and** priced by its own registry note at 43 h per arm-seed — more than a
week of free quota for one arm. Add a conjunct to `coverage.py` (it already owns
`root_dead()` and the CLAIM-DEAD predicate): when a spec's `PARKED:` marker
names a release spec, resolve that spec and report `PARK-ON-AN-UNREACHABLE-RELEASE`
if it is unreachable, welded, or foreclosed. Ratchet it as a total with the
existing classes per the `T0.31` precedent. Build the class RED-first and verify
it fires on `BA.02` before migrating anything.

**B4 — add up the diagnoses (section 8).** Six specs have independently recorded
that the venue, not the instrument, is what failed; nine queue rows carry nine
separate repairs. Before the 09-06 docket, write **one** consolidated note into
the `w0-too-shallow` row quoting all six recorded diagnoses verbatim with their
numbers, so Sunday's desk sees the convergence rather than six unrelated arm
choices. This is a bundling of *evidence*, not of decisions — the 09-06 stagger's
own distinction — and it does not pre-empt any of the nine.

**B5 — put the top blocker's age on the board (FINDING 2).** `run blocked` ranks
by `frees N`; nothing ranks by how long a blocker has been the top blocker.
`T2.01` has been rank 1 with an implementation untouched since **2026-08-09**.
Print `unchanged for N days` beside each terminal blocker, derived from the last
commit touching its implementation. Cheap, and it makes a 24-day stall
impossible to read as routine.

---

## FOR THE OWNER

**1. `D16` fires in three days and its premise moved today. You may want to look
before it does.** The guard is `T0.27` — the spec that stands between this
project and a silently-moved threshold. It is deliberately RED pending your
decision. Since the entry was written, its violation count has gone **1 → 2 → 3**,
the newest arriving **today**, and each was created by ordinary, honest work
following the loop's own documented procedure — implement, run, fix the code,
re-run.

The default (b) — *keep the red, touch nothing* — **still weakens nothing and is
still defensible**, and I am not asking you to change it. Two things are worth
knowing before it fires:

- Option (a)'s assumption that the pair will *"age out soon-ish"* now has a
  measurement against it. Violations are arriving at roughly one per 1.5 days
  against a 20-entry history. **The red will not clear itself.**
- The instrument tells all three violations with one sentence — *"that
  implementation was never committed"* — and it is **false for two of the
  three**. The failing bytes are preserved and cryptographically verified for
  `LG.00` and `T0.29`; the `git diff` the rule demands is possible for both.
  Only `T0.17` is truly lost.

Whether a verified preserved manifest should be a second lane is already routed
to the Review (`t027-preserved-failimpl-as-artifact`, DUE 09-05, the same day).
**No agent here has relaxed anything, and none should** — the builder correctly
escalated rather than repairing the guard that flags it, on the grounds that
conduct is class 3 and not its to measure. That was the right instinct and it
still is.

**2. Your two most GOAL-central specs are both FAIL and have been for weeks.**
`T2.01` (*locomotion beats a random policy*) blocks **38 specs** — the whole
curiosity family, the whole unison family, seven Tier-5 claims — and its
implementation has not been touched since **2026-08-09**. `LT.01` (*the Ladder
Test* — the ladder and the apple, your own image of the goal) is FAIL and blocks
9. The unblock path for `T2.01` now runs through a gate-design decision due
09-06 and a re-run in W36. **It is owned and clocked, which it was not two days
ago.** But if you want one thing to go faster than the calendar, this is it, and
it is worth more than every other row on the board combined.

**3. Four commitments you named as constitutional currently have nothing
falsifiable behind them** — smell, balance, shelter/building, and thermal
(*"too cold kills him, too hot kills him"*). Every park was individually honest
and evidence-backed; none was concealed. But *"he builds a shelter"* — your own
image of success — has no live claim, and balance's designated revival path is
blocked by arithmetic (FINDING 3). `five-commitments-are-claim-dead-behind-foreclosures`
is DUE 09-11.

**4. What the evidence is starting to say, offered as a reading and not a
decision.** Six specs, run independently, at different times, in different
families, have now each recorded that **the world — not the instrument — is what
failed**: `DP.04`, `SH.02`, `UB.14`, `BA.03`, `T3.09`, `LC.03`. The system has
routed nine separate repairs for what may be one cause. That question is
`w0-too-shallow`, DUE 09-06, and the builder correctly bundled the coupled
design rows onto that day. **If W0 is the common cause, then nine arm choices
and roughly 20 free GPU-hours a week are being spent on the wrong layer**, and
the cheaper answer is the world, not the arms. I am not ruling on it — that is
Sunday's desk, and the arms can be run, so rule 3 governs. But it is the single
judgment most likely to change how the next month goes.

**5. Nothing is waiting on you that shouldn't be.** Zero MEANS-ESCALATED, zero
UNDECLARED, zero OVERDUE. The eleven defaults armed for 08-31 all fired on 09-01,
by the builder, each naming a concrete artifact — I spot-checked four. The D1
disease is not present anywhere on this board.
