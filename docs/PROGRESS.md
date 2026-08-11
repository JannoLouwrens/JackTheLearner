# PROGRESS.md — the Review's current-state page

> Written by the Review organ. **Current state, not a log** — each run rewrites
> this file. The running history is `docs/PROGRESS_LOG.md`.
> Mode: DAILY (Part 2, the test re-examination, runs Sundays only).

**2026-08-11 17:02 UTC — DAILY. Window: 2026-08-10 17:00 → 2026-08-11 17:00.**

*The one sentence that explains this whole window: **the system was switched off
for 23 of its 24 hours.** Every organ hit the 90% weekly-usage gate at 17:07
yesterday and stayed down until the owner resumed it at 15:57 today. Read every
number below through that.*

---

## 1. The numbers

**Ladder:** 66/162 demonstrated (40.7%). Yesterday's same-hour reading: 58/147
(39.5%).

| | this window | prior 24 h |
|---|---|---|
| spec runs recorded | 4 | 29 |
| PASS | 4 (SM.01 + 3 re-certs) | 26 |
| FAIL / VOID / ERROR | 0 / 0 / 0 | 2 / 1 / 0 |
| net new demonstrated | **+1** (65 → 66) | +16 |
| registry growth | **+0** (162 → 162) | +42 (105 → 147) |

Three of the four runs were re-certifications (T0.13, T0.20, T0.21 re-run from
a clean tree). **One new capability entered the ladder: SM.01 — Jack has a nose.**

**Goodhart check — and this is the one number to not believe.** Pass rate rose
39.5% → 40.7%. It rose because the registry did not move at all while one spec
passed. A rate that improves during an outage is an artefact of the outage, not
a signal about the ladder. Say it plainly: **this window produced no evidence
either way about whether the ladder is outrunning its runner.** The last real
reading stands — 08-10's finding that the registry sprint had stopped
outrunning the runner — and it will not be re-testable until the builder has
run a full unimpeded day.

**Rework: 31 of 69 entries carry `attempt` > 1 (44.9%), up from 35.5%.** The
denominator barely moved (62 → 69) so this is a real rise, and it is still
concentrated in the harness: T0.13 at **20** attempts, LC.02 at 9, PG.6 /
T0.08 / T0.20 / T0.21 at 5 each. T0.13 ("no gate in the ladder is decorative")
having consumed twenty attempts is worth someone's attention on Sunday — a
spec that expensive to make true is either measuring something genuinely hard
or is badly shaped.

**Builder throughput — 24 scheduled fires, 1 iteration:**
- **23 fires refused by the usage gate** (17:07 → 15:07), logging "STOPPED at
  90–92% weekly usage" every hour. The gate worked exactly as designed.
- 1 iteration ran (15:57 → 16:20) and produced SM.01 PASS. Clean rc=0.
- 1 fire skipped at 16:07 because that iteration was still running — correct.
- 1 iteration in flight now (17:02).

**The gate's missing half was built inside the window** (`b1db303`): the 90%
stop had been logging "paused until the owner resumes" for hours with no resume
path in existence — the message wrote a cheque the code could not cash. It now
has `usage_gate()` shared by all four organs, and an owner resume carries a
ceiling *and an expiry*. The expiry is the right design: an override with no
end is a deletion of the limit that nobody remembers making.

**Credits and GPU — the two clocks, and they now point the same way:**
- Claude, week:all-models at **93%**, resets **2026-08-12 12:00 UTC**. The
  owner's resume ceiling (100%) expires at the same moment. Fable is at 100%,
  so every iteration is falling back to Opus.
- Kaggle: 11.96 h of 30 h charged to 2026-W32. **~18 h remain and expire
  2026-08-16.** Nothing submitted since the push block lifted.

---

## 2. The frontier

Recomputed from `run blocked` against the live ledger: **67 of 162 specs are
unreachable.**

| terminal blocker | status | frees | blocks |
|---|---|---|---|
| **T2.01** Locomotion beats a random policy | **FAIL** | **26** | **36** |
| **LC.03** Which learning cores learn to survive at all | NOT_RUN | **7** | 7 |
| UB.9 Heard, not seen | NOT_RUN | 4 | 7 |
| T2.08 Curiosity drives coverage | NOT_RUN | 3 | 4 |
| T2.06 Language-action alignment | NOT_RUN | 3 | 3 |
| T2.03 Pretrained vision features | NOT_RUN | 2 | 11 |

**THE BLOCKING FACT OF THIS REVIEW: D3 is answered and the queue in front of it
is untouched.** The owner said YES to `git push` on 2026-08-10. `origin/main`
and `HEAD` are now identical — **0 unpushed commits.** `gpu.py:assert_ref_is_current`
has nothing left to refuse. The first four entries of `run next` are
**T1.02 (gpu<20min), T2.01 (gpu<8h), T2.02 (gpu<8h), T2.03 (gpu<20min)** — all
four impossible for three days, all four possible right now, and between them
they carry T2.01's 26 freed specs and the vision-encoder seat's entire defence.
Nothing has been submitted. The reason is not neglect: of the 24 hours since
the unblock, the builder was awake for 25 minutes.

**LC.03 is new at #2 and it is the largest non-GPU unblock in the project.**
PS.01 passed at 08:32 yesterday (attempt 3), which retired the blocker that had
stalled the learning-core bakeoff — the match that decides HOW JACK LEARNS.
`ladder_prompt.md §0` was still telling the builder that PS.01 was FAIL and the
unit of work; corrected this run (§5). One caveat travels with it: `run stale`
lists PS.01 as a PASS recorded against code that has since changed, so LC.03
would be built on a stale certificate. PS.01 is CPU; re-run it first.

**Effort vs GOAL.md's path.** Too small a window to judge honestly. Four
commits: one built a sense (SM.01, the creature), one built the resume gate
(the machine), two were re-certifications and journal. I decline to compute a
ratio from four commits.

---

## 3. The honest paragraph

*(Not required in DAILY mode; three sentences because the window has one.)*
The most important thing that happened is that a door opened and nobody walked
through it — the push authorisation the last three reviews all escalated came
back YES, and the ladder's largest blocker has sat in front of an open door
since, because the loop was asleep for almost the entire time. The single best
step toward Jack was small and real: he can smell, and the certificate that says
so was gated in both directions, so it could have failed. The drift to watch is
not effort going to the wrong place this week — it is that the project's binding
constraint has quietly stopped being *what we know how to test* and become *how
many credits are left*, and no organ is scheduled to notice that.

---

## 4. REWRITTEN / STRENGTHENED

None. DAILY mode does not re-examine tests — Part 2 runs Sundays. Nothing was
weakened, no threshold moved, no control softened, no ledger entry touched.

Queued for Sunday's Part 2, so it is not lost: **T0.13** (20 attempts to make
true — re-examine its shape) and **SM.01's 0.33 intermittency shortfall**,
which the builder reported honestly and left ungated because it was outside the
registered hypothesis. That was the correct call at the time; whether SM.02 may
be built on top of it is a Part 2 question.

---

## 5. Steering maintenance performed

**`scripts/ladder_prompt.md` — four corrections, all of them things the hourly
builder reads before every iteration:**

1. **A live contradiction, and the expensive one.** §2 said *"D3 in
   DECISIONS_NEEDED.md is still OPEN… if it is non-zero the job cannot run, and
   escalating that is the useful iteration, not attempting it"* — while the
   section 90 lines below reads *"YOU MAY `git push`. Owner answered D3 on
   2026-08-10: YES."* A fresh iteration reading top-to-bottom is told the GPU is
   impossible and told to push every iteration. Following the first would burn a
   whole iteration re-escalating a settled decision, with the ladder's top
   blocker on the other side of it. Corrected to: if it is non-zero, **push**.
2. **§0 was pointing at spent work — the third occurrence of this failure
   mode.** It named PS.01 = FAIL as the unit of work; PS.01 has passed since
   yesterday morning. Re-aimed at LC.03, which `run blocked` now ranks second,
   with the stale-certificate caveat attached.
3. **§0aa** called UB.9 "second in the project"; it is third now that LC.03 is
   reachable. Corrected, with an instruction to read the live ranking.
4. **A cached count in the STANDING RULE** — *"Fifteen commitments read 0 PASS
   today"*; `run coverage` reads 12. The file's own opening rule is that state
   lives in the ledger, not here. Replaced with a pointer to the command.

**Added** to §2: the two-clocks directive. Kaggle hours expire unspent and
Claude credits are now the binding resource, so a GPU submission — a handful of
tool calls buying hours of compute — is the highest value-per-credit iteration
available. When credits are scarce, submit; do not spend the last iterations
writing CPU specs.

**Organ liveness — all five organs alive, none silent:**

| organ | cadence | last fire | verdict |
|---|---|---|---|
| builder | hourly | 17:02 (running) | OK — 23 gated fires, all logged |
| overseer | 6-hourly | 17:02 (running) | OK — 3 gated fires, all logged |
| field watch | Mondays | 17:02 (running) | OK |
| Review | daily | this run | OK |
| tmp_reaper | 4-hourly | 16:13 | OK |

**No organ was silent.** Every skipped fire wrote a line saying why, which is
the distinction this check exists to make: a gated organ is not a dead one.
Worth stating for the record because 23 consecutive skipped fires is exactly
what a dead cron looks like from the outside, and the only reason it did not
read that way is that the gate logs.

**Field watch:** `docs/FIELD_WATCH.md` is unchanged since the 2026-08-10 sweep,
already consumed — N1/N2/N4 are ACCEPTED entries in `INTEGRATION_QUEUE.md` and
N3 is marked REJECTED with its NE.03 re-open trigger. Nothing new to convert. A
fresh sweep is running as this is written; **the next Review must consume it.**

**Seat staleness (`docs/CHAMPIONS.md` rule 4) — three findings:**

- **Vision encoder** — `DEFAULT, never defended`. Its arena T2.03 is
  `gpu<20min` and was behind D3. **D3 is answered; the seat is twenty GPU-minutes
  from its first defence.** Carry one caution: T2.03 as written probes *frozen*
  DINOv2/SigLIP, and the PLASTIC-ONLY decree forbids seating a frozen component
  inside Jack. Run it as a measurement of how good our from-scratch features
  are, not as a seating contest — otherwise its winner cannot take the seat.
- **Learning core** — still `DEFAULT, never defended`, challenger "match in
  progress". Yesterday the finding was that the challengers could not reach the
  ring. Today the ring is open (PS.01 passed) and **still empty** — LC.03 is
  NOT_RUN a day later. The blocker is gone; the entry is the work.
- **Smell** — the seat text still reads "SM.01 is CPU and runnable today". It
  passed today. Reported rather than edited: CHAMPIONS.md edits are a Sunday
  ANATOMY AUDIT power, not a daily one.

---

## 6. FOR THE BUILDER

Ordered, and the order is dictated by two expiring resources. **Items 1–3 are
GPU submissions that cost few credits and buy hours of compute.**

1. **T1.02 — submit it. `gpu<20min`, first entry in `run next`.** It is an
   ERROR from an infrastructure fault, not a measurement, and it is one of only
   four specs behind the `generality` commitment (0 passing). Cheapest possible
   proof that the GPU path works end-to-end now that pushes are authorised —
   and if it fails honestly it kills the premise that this architecture can
   learn a state→action mapping at all, which is worth knowing before spending
   8 hours on T2.01.
2. **T2.01 — submit it. `gpu<8h`, frees 26, blocks 36, the largest single
   blocker in the ladder.** It is FAIL with a real number (every seed beat
   random, none by the pre-registered 5σ). Do not touch that threshold. Fits
   inside the ~18 h remaining, but only if it goes before 2026-08-16.
3. **T2.03 — `gpu<20min`, the vision-encoder seat's entire defence.** Read the
   PLASTIC-ONLY caution in §5 before you run it.
4. **LC.03 — the biggest non-GPU unblock (frees 7).** Re-run PS.01 first to
   clear its stale flag, or say in the commit that you did not. This restarts
   the match that decides HOW JACK LEARNS, and it runs beside any GPU job.
5. **The `EXIT` trap in `scripts/ladder_loop.sh`** — third review running that
   this is asked for. The usage gate proved the principle in the best possible
   way this window: 23 skipped fires were legible *only* because something wrote
   a line. Silence must never read as success.
6. Carried from 08-10, unchanged and still open: **N1 (the certificate pre-gate
   for UB.11)** before UB.9's results start feeding the ablation matrix, and the
   **LESSONS.md entry the scout could not write itself** — *confirm the results
   table says what the abstract says* (an AI-authored preprint claimed +34% over
   RND when its own table said ~25%, and it survived four of five checks).

---

## 7. FOR THE OWNER

**One item, and it is new. Credits are now the binding constraint on this
project, and nothing measures that.**

D3 is answered — thank you; it worked, `origin/main` is current and the GPU
path is open. But the resource that replaced it is worse, because it is
unmetered by any organ:

- The system spent **23 of the last 24 hours switched off**, at 90% of weekly
  Claude usage. It is at 93% now against a resume ceiling that expires
  **2026-08-12 12:00 UTC** — the same moment the week resets.
- Fable is at 100%, so every iteration falls back to Opus, at Opus's burn rate.
  The cheapest organ is running on the most expensive model, by fallback rather
  than by choice.
- Meanwhile **~18 Kaggle GPU-hours expire 2026-08-16** and nothing has been
  submitted. Those hours cost no Claude credits to *use*; they cost a few tool
  calls to *submit*. Right now the project is short of the resource it cannot
  buy and sitting on the one it cannot save.

**My recommendation, in priority order:**

1. **Spend the remaining credits on submissions, not on new CPU specs.** I have
   written that into `ladder_prompt.md §2` as a directive; it is operational and
   within my jurisdiction, and I am telling you because it changes what the
   builder will do with your last 7%.
2. **Consider a per-organ credit budget rather than one shared 90% cliff.** The
   current gate is all-or-nothing: at 90% the builder, the overseer, the scout
   and the Review all stop together. That is the wrong failure shape — it means
   the moment the project most needs its auditor is the moment the auditor is
   guaranteed to be offline. A cheap version: let the Review and overseer keep
   a small reserved slice.
3. **`DECISIONS_NEEDED.md` already carries "Claude credits are the binding
   resource and are unmetered (OPEN, owner)".** It was written before this
   happened. This window is its first real cost, and it should be re-priced with
   that evidence rather than left as a standing worry.

*Nothing in this review touched a threshold, a control, or the ledger. The four
steering edits are operational and are itemised in §5.*
