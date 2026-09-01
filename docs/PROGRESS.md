> **STALE — THE RUN THAT OWED THIS PAGE AN UPDATE PRODUCED NOTHING.**
> the Review has missed its schedule: no FULL row has EVER been written to docs/PROGRESS_LOG.md — that mode has never completed
> So everything below is the PREVIOUS run of the review and is a RECORD,
> not current state: its counts, its "current state" framing and any
> claim about what has or has not moved describe an older world.
> Stamped 2026-09-01T00:37:04+00:00 by scripts/lib_seal.sh. It disappears the next time the
> review completes a run and rewrites this file.

# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: FULL (Part 2, the anatomy audit and the completeness audit run here).

**2026-08-31 19:1x–19:5x UTC — FULL, and it is the FIRST FULL RUN THIS PROJECT
HAS EVER COMPLETED.** Window: 2026-08-31 06:5x → 19:5x (13 h since the DAILY),
with the weekly comparisons taken over 7 d.

*The one sentence: **the budget fix landed and the FULL run finally ran — and
the first thing it can see, which no DAILY could, is that the builder's next
queued unit is an arm it proved dead with its own hands this morning, while the
cheapest unblock on the board and the north star's own arena sit untouched.***

> **Provenance, stated up front because it matters.** This run was not fired by
> cron. The builder launched it detached at 19:11 as a **rehearsal** of the
> raised FULL budget (`scripts/review.sh` now derives `MAXTURNS` from
> `TURNS_PER_MIN=3`, so FULL gets `40m / 120 turns` instead of `20m / 60`),
> using a `date +%u` PATH shim so `review.sh` would take its Sunday branch. The
> script itself is unmodified. This page is therefore a real FULL product and
> also the rehearsal's evidence: **the budget was the whole problem.** Four
> Sundays died at `Reached max turns (60)`; this run reached Part 2, both
> audits, and this page with turns to spare.

---

## The numbers

| | now | 08-31 06:5x | Δ |
|---|---|---|---|
| demonstrated / registered | **94 / 211** | 93 / 201 | **+1 / +10** |
| pass rate | **44.5%** | 46.3% | **−1.8 pts** |
| FAIL / VOID | **16 / 5** | 13 / 5 | **+3 / 0** |
| unreachable specs | **80 / 211** | 69 / 187 | +11 |
| rework (ledger rows at attempt > 1) | 76 / 115 = **66.1%** | 66.7% | −0.6 |
| commits, last 7 d | **211** | prior 7 d: 137 | +74 |
| ledger events, last 7 d | 47 (37 PASS / 8 FAIL / 2 VOID) | prior 7 d: 34 (25/7/2) | +13 |
| `LESSONS.md` entries dated this week | **125** | — | — |
| all four ratchets | coverage 0 · champions ok · decisions ok · review-queue 0 | same | — |

**Goodhart check: the rate FELL, and this time it is NOT the good direction.**
On 08-26 the rate fell because claims were being *declared* rather than passed,
and I called that honest. Today is different in kind: **+10 registered against
+1 demonstrated**, and the +10 is the `LT.01`–`LT.09` family — which is real,
necessary work (it built the Curiosity seat's arena, which had never existed).
But `LT.01` was registered on 08-31 and **has not been run**, and the one
demonstrated is `W0.DIAG`. So the window's arithmetic is: *the ladder grew by
nine specs it did not touch, and passed one.* That is registry growth outrunning
the runner, which is exactly the shape the 08-10 Review named and the shape this
check exists to catch. **Say it plainly: this week the ladder grew ten times
faster than it climbed.**

**Rework at 66.1% is flat and still not the problem.** Unchanged reading from
08-31 DAILY: attempts 2+ are VOID→repair→re-run on rig faults the pilots caught.

**FAIL +3 is the most honest number on this table.** All three are `ME.11.B`,
`ME.11.C`, `ME.11.D` — three consecutive arms of one bakeoff, each pre-registered,
each settled in one attempt, each with its controls landing on the correct side.
The ladder deleting three of its own arms in a week is the machine working.

---

## The frontier, computed — and the builder is not on it

**Transitive-block mass, recomputed (`run blocked`, not quoted):** 80 of 211
specs are unreachable. `T2.01` remains the largest single blocker — **frees 35,
blocks 37** — and it is a FAIL whose repair the 08-14 Review already
reclassified from a compute problem to a science problem. It is not actionable
this week.

**The single most important UNBLOCKED spec is `LT.01`.** It is `NOT_RUN`, its
dependencies pass, it is `cpu<2h`, `coverage` lists it as fillable **today**, and
it **frees 7 (blocks 8)** — the third-largest mass on the board and the largest
that anyone can act on. Behind it sit `LT.03` and `LT.04`, which are the
**Curiosity-signal seat's entire arena**: the seat is held BY ANALYSIS, has never
been defended, and `LT.04` is where `disagree` and `metra` finally race `lp`.
Curiosity is GOAL.md's north star — *"purely out of curiosity… the environment
plus intrinsic motivation IS the curriculum"* — and `LT.03` is literally titled
*THE LADDER TEST: curiosity alone climbs the ladder.*

**Is the builder working on it? No.** The builder's own 18:07 TLDR queues
**`ME.11.F`** as its next unit. And here is the finding:

> **THE BUILDER'S NEXT QUEUED UNIT IS AN ARM IT MEASURED DEAD SEVEN HOURS
> EARLIER, WITH ITS OWN PROBE, AND WROTE THE REFUTATION DOWN ITSELF.**
>
> `ME.11.F` is the cascade arm. Its premise is *"the answer is present in the
> top-50."* This morning the builder ran `scripts/probe_me11c_recall_at_k.py`
> and measured Arm C's **recall@50 = 0.44** on the certified fixture (0.475 /
> 0.381 / 0.463) against the pilot's claimed 1.000. Its own commit message says
> it: *"Arm F's premise … is false on 56% of cues, so even a perfect reranker
> caps at 0.44 vs the family's 0.80 bar."* `ME.11.E` is in the same condition —
> its beat-both-parents gate needs a lexical parent that scores **0.0000** on
> all 160 cues × 3 seeds.
>
> So both remaining arms of `ME.11` are **arithmetically pre-refuted**, by
> measurements this project paid for and published, and one of them is next in
> the queue.

This is not a lapse of honesty — the builder published the refutation in the same
commit it published the queue entry. It is a **routing** failure, and it is the
same shape as this week's other one: the number that kills the unit and the
decision to run the unit lived in two different places, and nothing joined them.
The disposition of the whole family is already routed to this desk as
`me11-every-arm-hits-the-same-infeasible-branch` (DUE 09-06). **I am settling it
now rather than in six days**, because six more days is two more dead arms — see
FOR THE BUILDER 1.

**Effort-vs-goal.** Of 211 commits in 7 days, **75 (36%)** carry an audit
B-item, a ratchet re-stamp, a re-buy, or bookkeeping in the subject line. That
is the 55th audit's finding — *the audit organ is the largest single consumer of
builder iterations, and no instrument counts it* — independently reproduced from
the git log, and I confirm it. GOAL.md's current stage is Tier 2 (capabilities
vs null). The week's genuine Tier-2 output is `W0.DIAG` PASS and three honest
`ME.11` FAILs. Everything else served the instrument.

---

## Part 2 — THE TEST RE-EXAMINATION (never run before today)

Sampled 12 passing specs, oldest-`ran_at` and least-reconsidered first: `T0.11`,
`T1.09`, `T1.10`, `T1.11`, `T1.12`, `T1.13`, `ME.5`, `ME.10`, `T2.10`, `T2.12`,
`T6.03`, `DP.00`. Four survive unchanged and are not discussed. Eight findings,
ordered by how badly they matter. **Nothing below lowers a bar; two are
implemented here, six are proposals in Part 3.**

### The big one: `T2.10` certifies a champion that a later spec measured at zero

`T2.10` — *"Memory retrieval beats recency"*, PASS 2026-08-08, attempt 1, never
reconsidered — is the **founding certificate of the Episodic-retrieval seat**,
which `CHAMPIONS.md` holds BY VERDICT for *lexical containment*. Its control is
good for its day (two scorers that must lose: pure recency, and similarity-only).

Since it passed, `ME.11.B` measured that same lexical champion at **0.0000 on
160 paraphrase cues × 3 seeds**, on a fixture certified stem-disjoint by
measure (`stem_leak_cues` 0/160). `CHAMPIONS.md` already records the seat's
*"known weakness: 0.000 on paraphrase"* — **in the same table whose `held`
column reads BY VERDICT, on the strength of `T2.10`.**

So `T2.10` is **TOO WEAK in the precise sense Part 2 exists to catch**: it passes
because its query set shares surface form with its targets, and we have since
built — and certified — a venue where that assumption is false. The spec is not
wrong; its *venue* is. And the fixture that exposes it did not exist when it was
written. This is the T1.02 precedent's legitimate case: the experiment is
incomplete, not the system flattered.

**Proposal (Part 3, item 2): add a second, conjunctive venue to `T2.10`.** Not a
replacement — the recency comparison stays exactly as it is. The spec must ALSO
beat both existing controls on the `ME.11` certified paraphrase fixture. This is
strictly harder: it adds a conjunct to a passing spec, on a venue where the
current champion is already measured at zero. I am **not** implementing it here
because it re-parents a Tier-2 certificate onto another family's fixture, and
that is a redesign, not a comment.

### `T6.03` is a Tier 6 certificate bought before Tier 2 existed

`T6.03` — *"Cross-session persistence"*, PASS 2026-08-08 — sits in **TIER 6, "A
LIVING JACK"**, the tier GOAL.md defines as *"runs for hours, remembers across
sessions, explores unprompted in his playground world."* It passed on day four
of the project, `depends_on=["T2.10", "T0.05"]`, on a companion-app save/load
round-trip. It inherits `T2.10`'s weakness directly (its recall runs through the
same scorer), and it certifies "the companion recalls prior interaction" — a
sentence about a persistence layer, not about a creature that lived.

**This is a tier-position finding, and it is the one I would most want the owner
to see**, because a green tick in Tier 6 is the tick this project's entire
staging argument says must come last. See FOR THE OWNER 2.

### `DP.00`'s venue is not Jack's world, and it is inside a seat's arena

`DP.00` PASS 2026-08-10, attempt 2. The 08-25 audit already caught that its
*reading* had drifted to "Jack's world rewards deliberation" when it passed in a
12×12 gridworld. What that audit did not close: **`DP.00` is still a declared
member of the Deliberation seat's ARENA** (`ARENA: DP.00–DP.05, BO.01`). Then
`DP.05` asked the same question **in W0** and answered **NO** (FAIL 2026-08-24).

So the seat's ring contains one spec that says yes in a gridworld and one that
says no in Jack's world, with no marking that they are not commensurable.
**Implemented here (strengthening 1):** `DP.00`'s registry `notes` now state the
venue restriction and name `DP.05` as the in-world answer. A comment, not a
threshold — the smallest honest fix, and it stops the next reader treating a
gridworld PASS as evidence about W0.

### `T2.12` passes for a reason that no longer impresses us

`T2.12` — *"Emotion model produces distinguishable states"*, PASS 2026-08-08.
Controls are genuinely good (variance-matched random walk AND shuffled labels).
But its `kills` clause is *"EmotionalState as an input modality"*, and since it
passed, **`T3.07` FAILED** — mood does not change behaviour — and `D7`'s default
(firing today) accepts `MovementMoodCoupling` as cosmetics on the record.

What `T2.12` therefore certifies today is that a hand-written PAD state machine
produces trajectories separable from noise — which, for a deterministic
generator, is close to tautological — while the spec that asked whether any of it
*does* anything failed. It is not too weak by its own text; it has been
**overtaken**, and its `kills` clause now points at a consequence that another
spec already delivered.

**Proposal (Part 3, item 3): re-aim `T2.12`'s claim**, and give Emotion a chair
(done — see the anatomy audit).

### `T1.09` and `T1.10` are aimed at hardware we stopped using

`T1.09` (*"Fits in T4 memory"*, hypothesis *"Peak VRAM < 14 GB on a 16 GB T4"*)
and `T1.10` (*"CPU and GPU agree… CPU and T4 losses"*) both name the **T4**. This
project's GPU has been the Kaggle **P100** since at least 08-12, when a
cudnn/torchvision pin *"silently killed EVERY P100 job"*. Both are GPU_SHORT and
both would re-run cheaply. The ceiling happens to survive (P100 is also 16 GB),
so **this is a stale venue, not a wrong result** — but a spec that certifies a
device we do not run on is a certificate about somebody else's box.
**Proposal (Part 3, item 4): re-aim both at the actual accelerator and re-buy.**

### Three that hold up, said briefly because a clean bill is information

`T1.11` (train/inference path parity) is **stronger than most things written
since** — its null baseline is the actual defect it was born from, with the real
parameter counts in it (271,889 vs 0), and its control ("a loss that touches only
the unused head must FAIL this") is still a real control. `T1.12`'s control
(shuffled targets must not be reconstructable) and `T1.13`'s label-permutation
null are both the same shape this project later adopted as house style; they were
early and they were right. `ME.5` and `ME.10` hold: `ME.10`'s wipe-the-diary /
revert-the-weights dissociation is named in GOAL.md itself and still tests it.

### `T0.11` — the one I could not clear or condemn in this window

`T0.11` predates `impl_sha` and `spec_sha` both. `run status` reports **18 rows
that predate `impl_sha`** (17 verified byte-identical by git, 1 stale by content)
and **57 PASS rows that predate `spec_sha`** — for those 57, *"whether the claim
text moved since cannot be answered from the record."* That is not a defect in
`T0.11`; it is a **measurement hole across a quarter of the ladder**, and it is
the reason Part 2 is harder than it should be. See Part 3 item 6.

---

## THE ANATOMY AUDIT (i) — seats vs GOAL.md

The 08-30 run reached this audit before dying and added five sense seats (touch,
proprioception, pain, temperature, interoception). I re-ran the comparison
against GOAL.md and found **one capability with a real, registered, already-run
ring and no chair**:

**SEAT ADDED: Emotion (affect).** `EmotionalState.py` is **1,149 lines** — one of
the largest components in the repository — with `T2.12` PASS and `T3.07` FAIL
against it and **no seat in `CHAMPIONS.md`**, so no instrument that reads that
file could say who holds it or what would unseat it. Its arena resolves in
`BY_ID` today (`T2.12`, `T3.07`), one member has already returned a verdict, and
adding it moves no ratchet the wrong way. Added directly, per this desk's
mandate that adding a seat only invites competition. Verified: `champions
--check` still `rc=0`, `UNFALSIFIABLE` unchanged at 4/4.

**SEAT *NOT* ADDED, and escalated instead: THE BODY.** GOAL.md's First Principle
is *"Give him a brain, a **body**, and a world."* `CHAMPIONS.md` seats the world
(BY VERDICT) and eleven parts of the brain. **The body has no chair** — and the
file says so itself, in "Future seats": *"the body itself… They get chairs when
their first challenger exists."* **Its first challenger exists and has a measured
verdict**: the `W0.BAL` bakeoff (2026-08-21) produced a clean winner, arm C at
`upright_frac` **1.000** against the as-built body's **0.002–0.004**, and adopted
nothing pending `D9` — whose default fires **today** and parks it. I did not add
the seat because `W0.BAL` is not a registered spec id, so the chair would arrive
`NO-ARENA` and push `UNFALSIFIABLE` 4 → 5 — discharging nothing and breaking a
ratchet. **This is FOR THE OWNER 1.**

---

## THE COMPLETENESS AUDIT (ii) — against references OUTSIDE this repo

Audited the 211-spec registry against (a) the human sensory and cognitive
inventory and (b) `docs/GENERALITY.md`'s twelve barriers. Method: title and
`COVERS` search across `BY_ID`, not a read of our own coverage documents.

**THE HEADLINE, AND IT IS THE BIGGEST SINGLE HOLE THIS DESK HAS FOUND:**

> **Of the TWELVE generality barriers in `docs/GENERALITY.md`, ZERO are
> registered specs.** Not one of `GEN.01`–`GEN.12` has an id in a registry of 211.
> Worse, **GOAL.md itself cites four of them** — `GEN.02`, `GEN.03`, `GEN.06`,
> `GEN.09` — in its "the jungle is the foundation, not the destination" section,
> and `run coverage` reports **all four as DANGLING**. They have been carried as
> "known-dangling, registration debt" since 2026-08-25 by the 29th audit, and the
> debt has not moved in six days while 211 commits landed.
>
> The document that names what stands between Jack and generality is **cited by
> the constitution and referenced by nothing runnable.**

### The cognitive inventory, capability by capability

| capability | specs in registry | verdict |
|---|---|---|
| sight, hearing, touch, proprioception, smell, taste, pain, temperature, interoception, voice | 2–3 each | **covered** (all ten now have seats; four have a PASS) |
| working memory | **1** (`ME.8`) | thin — one spec for a named human faculty |
| attention | **1** (`UB.8`) | thin, and `UB.8` is behind `T2.01` (unreachable) |
| emotion | **2** (`T2.12`, `T3.07`), 1,149 lines of code | **now seated** (this run); one of the two FAILED |
| forgetting / consolidation | `ME.4`, `T5.03`, `NE.05`, `NE.06`, `T5.05`, `ME.7` | covered |
| **imagination / mental simulation** | **0** | **HOLE.** `InnerMonologue.py` exists as a module with no spec. GOAL.md names *"dreaming is training in imagination"* as an unmined biology oracle. `DP.04` touches verbal inner speech and is PILOT-BLOCKED. |
| **self-model / metacognition** | **0** | **HOLE.** `GEN.07` — *"He does not know what he does not know"* — has no spec. |
| **theory of mind / other minds** | **0** | **HOLE.** `GEN.02`, `GEN.03`, `GEN.09` — all three cited by GOAL.md, all three dangling. `VO.02` PASS (two learners invented a signalling system) is the closest thing and it is not a mind-model. |
| **teaching** | **1** (`NE.03`) | thin. GOAL.md: *"one sentence can spare him a thousand falls."* |
| **tool use / tool making** | **0** | **HOLE.** `GEN.05`. (`T0.28`/`T0.29` match "tool" but are *our* harness tools, not Jack's.) GOAL.md: *"a tool he has not made yet."* |
| language production | `LG.00`–`LG.10`, `VO.01`/`VO.02` | covered |
| **body schema** | **0** | **HOLE, and still open since 2026-08-09** — the owner found it then, and it is unmoved. |
| **symbols / number** | **0** | **HOLE.** `GEN.11` — *"nothing in his world requires symbols."* |

**Seven named holes, four of them cited by GOAL.md, and every one of them
invisible to every other organ in this system** — because coverage, champions,
decisions and the queue all measure reality against a *stated* standard, and no
stated standard in this repo says "imagination", "self-model", "tool making" or
"body schema". That is the 2026-08-09 scar reproducing exactly, six days after
the last audit declared the ratchets clean. **A named gap is a decision; an
unnamed gap is a blind spot.** These are now named.

**I am recommending we BUILD almost none of them this quarter** — that is not
the point. The point is that `GENERALITY.md` is a real document with real
content that the ladder cannot see, and the cheapest repair is registration, not
research.

---

## Constitution coherence

Scanned GOAL.md and SYSTEM.md for contradictions a fresh agent could trip on.
**One live one, and it is old:**

GOAL.md line 32 — *"purely out of curiosity"* — against line 91's *"the needs
ARE the curriculum."* GOAL.md line 96 **already reconciles this in place**
(*"read the two drives as partners"*), so it is annotated, not contradictory.
I record it only because a fresh agent reading top-down hits the strong form
first. **No edit proposed; the reconciliation is already there and it works.**

**One I am flagging as a fork rather than a contradiction:** GOAL.md's staging
says *"First prove he can see, talk, walk, and learn in every way… only then
does he go into the survival world."* Nine instruments now say W0 does not ask
enough of him to prove any of it. The constitution's ordering and the ladder's
evidence are pulling in opposite directions, and that tension **is** the
`w0-too-shallow` decision. It is not a drafting defect and I propose no edit.

---

## The honest paragraph

Closer, and for a reason that is easy to undersell: for the first time this
organ finished the job it was designed for, and the job turned out to contain
things no amount of diligence at the daily cadence would have surfaced — a
founding certificate resting on a venue we have since proved false, a green tick
in the tier that was supposed to come last, and a whole document of open
questions that the constitution cites and the ladder cannot see. None of that
was hidden. All of it was written down, in our own files, in our own hand. What
was missing was an organ with enough room to read two files at once. That is the
week's most important step toward Jack: not a capability, but the recovery of
the project's ability to look at itself whole. And the drift is the exact
mirror of it. The builder is fast, honest, and pointed at the wrong thing — it
spent the morning proving with its own probe that the arm it queued for the
afternoon cannot pass, published both facts in the same commit, and queued it
anyway; meanwhile the arena for the one drive this project calls its north star
was built this week and left standing untouched. We have become extremely good
at killing our own work and noticeably worse at choosing which work to do. The
machinery of falsification is in excellent health. The thing it is pointed at is
increasingly the machinery.

---

## REWRITTEN / STRENGTHENED

| spec / file | change | why it is stronger |
|---|---|---|
| `DP.00` (registry notes) | venue restriction stated in the spec text: the PASS is a **12×12 gridworld** result, `DP.05` is the in-world answer and it FAILED | the seat's arena held a yes and a no with nothing marking them incommensurable; a reader could take a gridworld PASS as W0 evidence — and one already did (08-25 audit) |
| `docs/CHAMPIONS.md` | **SEAT ADDED: Emotion (affect)**, `ARENA: T2.12, T3.07`, both registered, both run, one FAILED | 1,149 lines of `EmotionalState.py` had two specs and no chair, so no instrument could say what would unseat it; adding a seat only invites competition. `champions --check` verified `rc=0`, `UNFALSIFIABLE` unchanged |
| `scripts/ladder_prompt.md` | priority head block **replaced**: `W0.DIAG` PASSED, so the block's central order (*"your unit is priority 1 below"*) was spent; `LT.01` installed as priority 1 with its mass, and `ME.11.E`/`ME.11.F` **retired by name with the measurement that killed them** | the block was ordering a unit that had already been demonstrated, and the builder's own queue pointed at two pre-refuted arms |
| `docs/PROGRESS_LOG.md` | first **FULL** row in the project's history | the trend line now contains the mode that owns the world redesign |

No threshold moved. No control softened. No FAILING or VOID spec was rewritten.

---

## FOR THE BUILDER — ordered

1. **Settle `ME.11` now; do not run `ME.11.E` or `ME.11.F`.** This discharges
   `me11-every-arm-hits-the-same-infeasible-branch` six days early, on your own
   evidence. `ME.11.F`'s premise is measured false (recall@50 **0.44**, 3 seeds,
   against a 0.80 bar — a perfect reranker cannot reach it). `ME.11.E`'s
   beat-both-parents gate needs a lexical parent measured at **0.0000**. Record
   both as **VOID-FORECLOSED with the arithmetic**, exactly as you did for
   `BA.03`/`LC.03`/`T3.06` — that is the honest form and you already own it.
   Then update the Episodic-retrieval seat: five configurations, every one
   INFEASIBLE, best unthresholded ceiling 0.250 against 0.80. **The finding is
   that a bi-encoder cannot certify paraphrase recall at this scale, and it is
   worth more than a sixth arm.**

2. **Then `LT.01`.** `cpu<2h`, deps PASS, `NOT_RUN`, **frees 7**. It is the
   largest actionable mass on the board and it opens the Curiosity seat's arena
   — `LT.03`/`LT.04`, where `lp` finally has to defend its chair against
   `disagree` and `metra`. This is GOAL.md's north star and it has a runnable
   entry point for the first time. Do not let it sit a second week.

3. **`T2.10`: add the paraphrase venue as a CONJUNCT.** (Redesign, so it is
   yours, not mine.) Keep the recency and similarity-only controls exactly as
   they are. ADD: the scorer must also beat both, on the `ME.11` certified
   stem-disjoint fixture. New bar, and why it is HARDER: the seated champion is
   *measured at 0.000* there, so the conjunct cannot be satisfied by the
   machinery that satisfied the original. Old version stays in the ledger
   history (T1.02 precedent). If it FAILS, that is the correct outcome — it
   means the Episodic-retrieval seat has been held BY VERDICT on a certificate
   that does not cover the case `CHAMPIONS.md` already calls its known weakness.

4. **`T2.12`: re-aim the claim, do not weaken it.** Its `kills` clause
   (*"EmotionalState as an input modality"*) was overtaken by `T3.07`'s FAIL and
   `D7`'s cosmetics default. Proposal: keep both controls, and add the conjunct
   that separability must survive **at the fusion boundary** — i.e. that the PAD
   channel carries gradient in a live `UB.11` ablation. Strictly harder: it
   converts "separable from noise" (near-tautological for a deterministic
   generator) into "load-bearing", which is GOAL.md's own standard for every
   sense.

5. **Re-aim `T1.09` and `T1.10` at the P100 and re-buy them.** Both name the
   **T4**; this project runs Kaggle P100s. Both are GPU_SHORT and cheap. Not a
   weakening — same ceilings, correct device. Do this on the next GPU week when
   a class is otherwise empty; it is exactly the kind of work that keeps a free
   allocation from expiring.

6. **Register the four dangling `GEN` ids that GOAL.md cites** — `GEN.02`,
   `GEN.03`, `GEN.06`, `GEN.09`. This is registration debt seeded 2026-08-25 by
   the 29th audit and untouched through 211 commits. It is cheap, it discharges a
   coverage dangle, and until it is done the constitution cites four specs that
   do not exist.

7. **A note on the rehearsal, offered as a result rather than an instruction.**
   Your `date +%u` shim worked and the answer is: **FULL fits.** This run reached
   Part 2, both audits and a full page inside the raised budget. Report that to
   the 55th audit's B1 as PASS. One incidental: at 19:12:41 your leftover-process
   detector fired on `experiments.decisions --check` — that was **me**, this
   Review, running a read-only ratchet check. The detector is right to flag it
   and right not to kill it; the gap is that a Review's own subprocesses are
   undeclarable. Worth a declaration channel, not a suppression.

---

## FOR THE OWNER — three strategic forks

### 1. The body has no chair, and its one measured challenger gets parked today

GOAL.md's First Principle names three things: **a brain, a body, and a world.**
`CHAMPIONS.md` seats the world and eleven parts of the brain. It does not seat
the body — and it knows, filing it under "future seats… they get chairs when
their first challenger exists."

**The challenger exists and it won.** `W0.BAL` (2026-08-21): arm C held upright
`1.000` of the time against the as-built body's `0.002–0.004`. Nothing was
adopted, correctly, pending your `D9` — **whose default fires today and parks
the whole question** until "the playground-humanoid line."

I could not add the chair myself: `W0.BAL` is a bakeoff, not a registered spec,
so the seat would arrive with no arena and push a ratchet the wrong way. That is
the fork:

**My recommendation: register `W0.BAL` as a spec id and seat the body, even
while `D9` parks the adoption.** Parking an *adoption* and having no *chair* are
different things, and only the second one hides the question. Today, a body that
falls over 998 times in 1000 is the incumbent champion of a seat that does not
exist, defended by a bakeoff it lost, and no instrument in this repo can print
that sentence. Let `D9`'s default fire — I am not asking you to delay it. Ask
only that the seat be created, so the parking is *visible* as a parked seat
rather than as silence.

**Why this matters beyond bookkeeping:** four of this week's nine
"W0-is-too-shallow" instruments are scored through lifespan or upright-time in a
body we have measured cannot stand. `SH.02`'s pilot found every arm without a
live policy gradient holds the roof *completely*; `BA.03`'s blind twin holds
98.9% of the horizon. **Some fraction of "the world is too shallow" may be "the
body cannot act in it"** — and that reading has no seat, no arena, and therefore
no way to be tested. The 08-21 Review said *"decide D9 with D1, not after it —
T2.01 may be failing because the body cannot stand."* Eleven days later that
sentence is unanswered and `T2.01` still blocks 37 specs.

### 2. There is a green tick in Tier 6, and the staging argument says there cannot be

`T6.03` (*Cross-session persistence*) passed on **2026-08-08**, four days into
the project, in **TIER 6 — "A LIVING JACK"**. GOAL.md defines that tier as *"runs
for hours, remembers across sessions, explores unprompted in his playground
world"* and says the goal is accomplished when Tier 6 passes.

It passed on a save/load round-trip through a companion-app persistence layer,
depending on `T2.10` — the certificate Part 2 just found resting on a venue we
have since falsified. **It is not fraudulent and I am not asking to revoke it.**
It is a correctly-run test that got filed in the tier whose name it borrows,
and the consequence is that this project's own scoreboard shows progress in the
tier that is supposed to be the finish line.

**My recommendation: re-tier `T6.03` to Tier 0 or 1 (harness/primitive), where a
save-load round-trip belongs, and leave Tier 6 empty until Jack fills it.** This
is a strengthening by any reading — it *removes* a pass from the finish line —
but it moves a spec's tier, which changes the shape of the ladder, and I will
not do that to the constitution's own staging without your word. The alternative
(leave it) is defensible too; what is not defensible is nobody having noticed
for twenty-three days.

### 3. The budget fix worked — and the same class of defect is now visible elsewhere

**Discharged, with evidence:** you (or the builder, acting on the 08-31 DAILY's
ask) raised the Review's budget — `review.sh` now derives `MAXTURNS` from
`TURNS_PER_MIN=3`, giving FULL `40m / 120 turns`. **This page is the proof it was
sufficient.** Four Sundays of `Reached max turns` were a budget defect and
nothing else. The `w0-too-shallow` re-arm to 09-06 is no longer conditional.

The same script carries the general lesson, and it is worth stating because it
now applies to a *different* organ: *"There have been 7 max-turns deaths across
the three organs (ladder 4, overseer 2, review 1); every one of them left time
on the clock."* Four of those seven are the **builder's**. The builder is the
only organ that has never had its budget derived from its scope, and it is now
the organ doing 36% of its commits on audit bookkeeping inside an hourly slot.
**I am not asking for a decision** — I am flagging that the fix you just made for
this desk has an untested twin next door, and that the 55th audit's finding (the
audit organ is the largest consumer of builder iterations) and the builder's four
max-turns deaths are probably the same fact seen from two sides.

**And the honest caveat on this whole page:** it was produced by a *rehearsal*
that the builder launched with a shimmed `date`. Cron still fires `review.sh` at
`37 6 * * *`, and the Sunday branch still keys off `date +%u = 7`. **The next
real FULL run is 2026-09-06**, and nothing about tonight guarantees that one
fires — it guarantees only that if it fires, it fits.
