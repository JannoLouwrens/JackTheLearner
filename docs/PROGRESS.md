# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-10 06:37 UTC — DAILY. Window: 2026-08-09 06:37 → 2026-08-10 06:37.**

*First creation of this file. `SYSTEM.md`'s map and `scripts/ladder_prompt.md`
step 5 have instructed every builder iteration to read
`docs/PROGRESS.md`, section FOR THE BUILDER, since 2026-08-09 — and the file
has never existed. 20 iterations followed a dangling pointer. That is the
smallest finding here and the most embarrassing one.*

---

## 1. The numbers

**Ladder:** 58/147 demonstrated (39.5%). Yesterday's same-hour reading: 42/105
(40.0%).

| | this window | prior 24 h |
|---|---|---|
| spec runs recorded | 29 | 22 |
| PASS | 26 | 21 |
| FAIL | 2 (T2.01, PS.01) | 0 |
| VOID | 1 (T2.02, restated) | 0 |
| ERROR | 0 | 1 |
| net new demonstrated | **+16** (42 → 58) | +9 |

**10 of the 26 PASSes were re-certifications, not new capability** (PG.1, PG.2,
PG.3, PG.5, T0.01, T0.06, T0.08, T0.12, T2.00, T2.20 re-ran against current
code). Net new capability is 16, and the honest denominator for "what did today
buy" is 16, not 26.

**Goodhart check — the good news, stated precisely.** Registry grew 105 → 147
(+40.0%); passes grew 42 → 58 (+38.1%). Pass rate moved 40.0% → 39.5%: a 0.5-point
dip against a 40% expansion of the ladder. This is *not* the 2026-08-07..09
pattern (40.0% → 38.3% while the ladder outran its own results). For the first
day on record the ladder passed at very nearly the rate it grew. Say it as
information: **the registry sprint of the last three days has stopped
outrunning the runner.**

**Rework:** 22 of 62 ledger entries carry `attempt` > 1 (35.5%). Concentrated,
not diffuse — T0.13 at 15 attempts, LC.02 at 7, T0.08 at 5, PG.6 at 4, T0.14 at
4. Every one of those is a Tier-0 harness spec. The ladder's rework is being
spent on the measuring instrument, which is the right place for it, but see §3.

**Builder throughput:** 23 scheduled fires, of which
- 1 **ABORT** at 11:07 — load 8.37 above the 6.0 ceiling, box left to the
  tenants. Correct behaviour, not a fault.
- 4 **rc=1** (07:07–10:07), all 2–5-second credit-exhaustion deaths on Fable.
  The opus fallback landed at 12:07 and has held for 17 consecutive iterations.
- 2 iterations (17:07, 22:07) **did the work, committed, and emitted no
  `iteration end` line.** Their output is in the log; the instrumentation is
  what failed. The overseer asked for an `EXIT` trap on 2026-08-09 18:48 so a
  killed iteration still records that it was killed; it is still unbuilt, and
  these two are what it would have caught.
- 16 clean rc=0.

**The GPU:** the first GPU *result* since 2026-08-06. `jack-ladder-1786304547`
charged 5.58 h to week 2026-W32 and returned T2.01 = **FAIL**. Not an ERROR,
not a VOID — a measurement.

---

## 2. The frontier

Recomputed from `run blocked` against the live ledger (not quoted from
DIRECTION_AUDIT): **63 of 147 specs are unreachable.**

| terminal blocker | status | frees | blocks |
|---|---|---|---|
| **T2.01** Locomotion beats a random policy | **FAIL** | **26** | **36** |
| UB.9 Heard, not seen | NOT_RUN | 4 | 7 |
| PS.01 The drive layer is a real control problem | **FAIL** | 4 | 4 |
| T2.08 Curiosity drives coverage | NOT_RUN | 3 | 4 |
| T2.06 Language-action alignment | NOT_RUN | 3 | 3 |
| T2.03 Pretrained vision features | NOT_RUN | 2 | 11 |

**T2.01 is the frontier and it is not a mystery any more.** Its recorded
metrics say `all_seeds_beat_random = 1.0` — every seed *did* beat random — but
not by the pre-registered 5 sigma, against an untrained control sitting at 0.68
sigma. The policy learns something and the something is small. That is a far
more useful result than the VOID it replaced, and it is the correct thing for
the ladder to be stuck on.

**Is the builder working on it? No — and it cannot.** T2.01's re-run is a GPU
job, `gpu.py:assert_ref_is_current` refuses any HEAD that is not an ancestor of
`origin/main`, and there are **19 unpushed commits**. That is D3, open since
2026-08-08. See FOR THE OWNER.

**Effort vs GOAL.md's path.** GOAL.md places us at stage 2, "Capabilities vs
null". Of 20 commits in the window, roughly 13 served **Tier 0** — the
measurement machine (T0.15–T0.20, ledger-writer integrity, the bakeoff gate
mode, organ scratch and the reaper). Four served Tier 2 capability (PG.6/PG.7,
the PS.01/J bakeoff). One registered the five missing senses. Two were
steering/scouting. **The majority of the week's effort went into the
instrument, not the creature.**

---

## 3. The honest paragraph

No numbers. The machine got sharply better at telling the truth this window,
and Jack did not get more alive. The ledger learned to tell a run from an edit,
to re-derive its own verdicts backwards from the record, to notice a stale
writer reverting entries behind its back, and to refuse a bakeoff whose arms
cannot separate a fall. Those are real and they are the kind of thing that
compounds. But the creature they measure still cannot walk, cannot smell,
cannot taste, cannot make a sound, and does not have a body schema — and the
single most important step of the window was the one that hurt: a GPU run came
back and said the locomotion policy beats random by less than we demanded. That
is the week's best moment, because it is the first time in four days that
reality got a vote. The most concerning drift is the mirror of it: we have
built an extraordinary apparatus for detecting self-deception and pointed most
of this window's effort at the apparatus. Thirteen of twenty commits improved
how we would know, and four improved what there is to know. The scar-driven
rule in SYSTEM.md says organs are earned by failure, and every one of these
was — but the corollary in the same paragraph is the one now coming due: when
the machine is sufficient, prove it by throughput. The instrument is
sufficient. Point it at him.

---

## 4. REWRITTEN / STRENGTHENED

None. DAILY mode does not re-examine tests — Part 2 runs Sundays, and daily
rewrites would churn the ladder. Nothing was weakened, no threshold moved, no
ledger entry touched.

Steering files corrected instead (operational, not constitutional — see §5).

---

## 5. Steering maintenance performed

**`scripts/ladder_prompt.md` — five stale directives corrected, all of them
things the hourly builder reads before every iteration:**

1. *"T2.01 and T2.02 are VOID"* — T2.01 re-ran and is now **FAIL** with a real
   number. Replaced with the measurement and its blocked-mass.
2. *§0aa "CHEAPEST HIGH-LEVERAGE WORK, DO IT FIRST: run PG.6 and PG.7"* — both
   PASS. The directive would have sent an iteration to re-run spent work.
   Re-aimed at UB.9, which `run blocked` now ranks second in the project.
3. *§0 "FINISH THE LEARNING-CORE BAKEOFF… implement and run LC.02…LC.06"* —
   LC.02 PASSES and **LC.03–LC.06 are all blocked behind PS.01 = FAIL**. The
   directive named the highest-leverage unblocked work in the project and every
   item on its list was either done or unreachable. Re-aimed at PS.01, with the
   T1.02 precedent attached so the re-derivation cannot become a weakening.
4. *§2 Kaggle calendar* — cached "assume ~0h left, the FIRST Kaggle job is
   T2.02". Both false. Replaced with a pointer to `gpu_budget.json` (live
   source, per the LESSONS rule that priorities never cache state) and an
   explicit pre-flight: check `origin/main..HEAD` before planning GPU work,
   because an unpushed HEAD makes the job impossible.
5. **A live contradiction.** "Settled decisions" states the PLASTIC-ONLY decree;
   forty lines later "Context worth carrying" instructs the builder to *"freeze
   a pretrained trunk and learn a small adapter"*. A fresh agent reading
   top-to-bottom is told both. Marked superseded, and the part of that finding
   that still binds (there is no data for a bespoke 105M brain) kept and
   pointed at PL.00's feasibility check.

**Organ liveness — all four organs alive, none past cadence:**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly (`7 * * * *`) | 06:17 | OK |
| overseer | 6-hourly (`37 */6`) | 06:37 (running now) | OK |
| field watch | Mondays (`37 5 * * 1`) | 05:37 today | OK |
| Review | daily (`37 6 * * *`) | this run | OK |
| tmp_reaper | 4-hourly | 04:13 | OK |

**Seat staleness (`docs/CHAMPIONS.md` rule 4) — two findings:**

- **Learning core** is marked `DEFAULT, never defended` with challenger status
  *"match in progress"*. The match is **not** in progress: its arena is
  LC.00–LC.06 and LC.03–LC.06 are blocked behind PS.01 = FAIL. A default
  champion whose challengers cannot reach the ring is the exact condition rule
  4 names. PS.01 is the unblock, and it is CPU.
- **Vision encoder** is `DEFAULT, never defended`; its arena T2.03 is
  registered, NOT_RUN, `gpu<20min` — twenty GPU-minutes from being defended,
  behind D3.

**Field watch consumed** (`docs/FIELD_WATCH.md`, sweep of 2026-08-10, 4
nominations). Dispositions are in FOR THE BUILDER; N3 is rejected below.

---

## 6. FOR THE BUILDER

Ordered. Items 1–3 are queue work; 4–6 are the field watch converted.

1. **PS.01 — re-derive the threshold from the measured constants and re-run.**
   Highest-leverage *unblocked* work in the project: it frees LC.03–LC.06 and
   thereby restarts the learning-core match, which decides HOW JACK LEARNS.
   The run measured J0 = 2.405 m/s and alpha = 0.0293 and refuted §2.3's energy
   arithmetic. **T1.02 precedent governs**: this is legitimate only because the
   *experiment's arithmetic* was wrong, never to make the ladder look better.
   The re-derived threshold must be stated in the commit with its derivation,
   and the old version stays in the ledger's history. If the re-derivation
   makes the gate *easier*, stop and escalate instead — that is the shape of a
   weakening.
2. **UB.9 "Heard, not seen"** — second in blocked-mass (frees 4, blocks 7) and
   the gate on the entire unison ladder, where **0 of 37 specs pass**. CPU.
   Before running it, read item 4.
3. **Build the `EXIT` trap in `scripts/ladder_loop.sh`.** The overseer asked
   for it 12 hours ago; two iterations in this window (17:07, 22:07) completed
   work and recorded no end line. Silence must never read as success — that is
   the same principle the organ-liveness check enforces one level up.
4. **N1 — the certificate pre-gate for UB.11 (accept; strongest nomination).**
   `UB.11` is the standing modality-ablation matrix whose verdict is *"deletion
   is the default action, not a discussion"*, and it has a placebo column but
   **no positive control**. As written it cannot distinguish "this sense is
   decorative" from "this fixture gave this sense nothing to say" — and it
   deletes the encoder either way. [arXiv:2607.27017] supplies a certificate
   that probes the raw observations first, proving a sense was recoverable
   before concluding the model ignored it. This is a *strengthening of an
   existing spec* and it gates a destructive action; take it before UB.9's
   results start feeding the matrix. ~5M params, <1 h.
5. **N2 — anti-collapse regularisers as `A4b`/`A4c` in LEARNING_CORE §5.4**
   (accept). SMWM [arXiv:2606.20104] and SIGReg/LeJEPA [arXiv:2511.08544] would
   delete A4's EMA target encoder. Enters as bakeoff arms, decided in the
   arena, never by argument.
6. **N4 — the entity-collision eval protocol for ME.11** (accept, with a
   caveat to carry). It floors BM25 by construction, which is the *opposite*
   discipline to our lexical-disjointness invariant. Register it as an
   eval-set design question, not as a result to adopt.
7. **A LESSONS.md candidate the scout could not write itself** (LESSONS.md is
   not the field watch's to edit): a `clawrxiv.io` preprint, self-described as
   *"published autonomously by AI agents"*, matched our open questions almost
   exactly and claimed +34% over RND when **its own results table says ~25%**.
   It survived four of the scout's five checks. The cheap check that catches it
   alone: **confirm the table says what the abstract says.** Generalise and
   append it.

**REJECTED — N3, interoceptive precision allocation** ([arXiv:2608.04232]).
One line of reason, per the queue protocol: SYSTEM.md's *no new organ without a
scar* — no failure in this repo has been traced to uniform interoceptive
precision, so there is nothing for it to fix yet. The scout argued against its
own nomination on exactly this ground and was right to. **Re-open trigger:**
NE.03 runs and the uniform nine-float design underperforms. Mark it REJECTED in
`INTEGRATION_QUEUE.md` with that trigger attached — a rejection without a
re-open condition is just forgetting.

---

## 7. FOR THE OWNER

**D3 — may the loop `git push`? This is the whole bottleneck and it is one
line.** *(Open since 2026-08-08; this is the third organ to escalate it.)*

- 19 commits sit unpushed. `gpu.py:assert_ref_is_current` correctly refuses to
  build a GPU job from a HEAD that GitHub does not have — the VM clones from
  GitHub, so unpushed work is simply not there.
- Behind it: **T2.01 = FAIL, which alone frees 26 specs and blocks 36** — the
  largest single blocker in the ladder — plus T2.03 (the vision-encoder seat's
  entire defence, 20 GPU-minutes) and T1.02.
- Week 2026-W32 has burned 11.96 h of the 30 h Kaggle quota. **~18 h remain and
  they expire 2026-08-16.** Unspent free quota is not saved; it is lost.
- Nothing new gets published by this. The repo is already public and already
  contains every file involved; these are ladder specs and harness fixes, the
  same category as the ~76 commits already on `origin/main`.

**My recommendation: option 2 — authorise pushes when a GPU submission requires
one.** It unblocks the frontier today, covers every case that has actually
arisen in three occurrences, and keeps the general "nothing leaves the box"
instinct intact. Option 1 (standing authorisation) is also fine and simpler.
Option 3 (keep it your call) is the current state by default rather than by
choice, and its price is roughly 18 GPU-hours and the top blocker staying
blocked for another week.

**Second, smaller: the Review had no output file for 20 builder iterations.**
`SYSTEM.md` and the builder prompt both direct every iteration to
`docs/PROGRESS.md § FOR THE BUILDER`; it did not exist until this commit. No
harm done that I can find — the builder read a missing file and moved on — but
it is worth knowing that a documented channel between two organs was dead and
no organ noticed. The overseer audits spec diffs and the ledger; nothing audits
whether the *pointers between organs* resolve. I am not proposing a new organ
for it (no scar yet, per SYSTEM.md); I am recording it so that if it happens a
second time, the scar exists.
