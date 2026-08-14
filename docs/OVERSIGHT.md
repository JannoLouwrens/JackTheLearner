# OVERSIGHT — 15th audit, 2026-08-14 00:40 UTC

## VERDICT: ON TRACK

The ledger itself is sound. All **79 PASS** records name a commit that still
exists, all have an implementation on disk, and **every PASS whose spec
declares a control both ran that control and reads it in `_check`** — I
verified the second half by AST over all 79, not by trusting the presence of
`control_metrics`. Only T0.01 and T0.10 lack control metrics, and both
correctly declare `control=None` (repo-imports-clean, Kaggle round-trip). No
threshold moved in the loosening direction in seven days. No drift from
GOAL.md. `coverage.py` exits 0.

Two findings. The first is that **the 14th audit's RANK 1 fix closed the four
instances and did not close the class** — the detector built to close it has an
empty domain and reads zero for that reason, and an independent check finds
three records it calls "unchecked" that are in fact stale. The second is about
this document: **the 14th audit's own owner-facing note recommends an
architecture option that the repo's constitution forbids**, and pre-commits a
one-line trigger to enact it, 585 lines below an unanswered overseer entry
saying exactly that. That one damages the ledger least and is the most urgent
thing in this report.

**169 specs · 79 PASS · 3 FAIL · 2 VOID.** Builder: **24 iterations in 24 h,
21 rc=0, 75 → 79 PASS.**

---

## 0. Is the ladder the RIGHT ladder?

`python -m experiments.coverage` → **exit 0. Zero commitments with no declared
spec.** The 2026-08-10 miss has not recurred.

| tier | | |
|---|---|---|
| T0 harness | 28/28 | complete |
| T1 primitives | 13/13 | complete |
| T2 vs null | 36/59 | 1 FAIL (T2.01), 1 VOID (T2.02) |
| **T3 earn your parameters** | **0/14** | **1 FAIL (T3.07) — the tier's first entry ever** |
| **T4 unison** | **1/23** | 1 FAIL (T4.02) — the tier's second entry ever |
| **T5 THE CLAIMS** | **0/27** | 1 VOID (BA.02) |
| T6 living Jack | 1/5 | |

**Zero-pass constitutional commitments: 17 of 23**, unchanged for three audits.
Six commitments have a passing claim-kind spec: curiosity (1 of 12 specs),
hearing (1 of 6), one-brain/unison (1 of 21), generality (1 of 4),
memory-across-lives (1 of 3), damage (1 of 1). Everything else is fixtures and
sensors — real work, but not the commitment.

---

## RANK 1 — the staleness class was not closed. The detector built to close it has an empty domain, and it passed its own planted-positive test.

**No PASS is shown to be false by this. The damage is that three of them cannot
be shown to be true, and every instrument in the repo currently says they can.**

The 14th audit (B1) asked for two things: re-run the four unprotected
certificates, and *"close the class rather than the instance"* by splitting
`run stale`'s **"cannot be checked"** into **"cannot be checked"** and
**"cannot be checked AND its declared dependency has since moved"**. The
builder did both (`b5db2d4`, `5738b4d`): the four certificates re-ran and now
carry `impl_sha`, and T0.17 gained an `UNVERIFIABLE_MOVED` detector that
demonstrably fires on a fabricated moved-dependency record and spares an
unmoved one. `run stale` now reports:

    30 further entries predate `impl_sha` and cannot be checked at all
    (no declared dependency has moved)

**That parenthesis is true and empty. All 30 of the 30 declare no `IMPL_DEPS`
at all**, so `UNVERIFIABLE_MOVED` is not zero — it is undefined over an empty
domain:

| unstamped records | declare `IMPL_DEPS` | detector can protect |
|---|---|---|
| 30 | **0** | **0** |

`IMPL_DEPS` and `impl_sha` shipped together. Every record old enough to lack
the stamp is old enough to lack the declaration, so the detector's domain and
the population at risk are disjoint **by construction** — and will stay so
until each record is re-run, which is the same attrition argument the 14th
audit's own LESSONS entry (`docs/LESSONS.md:3454`) says will not work.

**The independent check the detector cannot make.** Provenance does not need a
declaration: for each unstamped record, compare its implementation file at HEAD
against the same file as it stood at the recorded run. That is exact, it needs
no opt-in, and it finds three genuine stale records:

| spec | status | ran | drift in its own implementation file since the run |
|---|---|---|---|
| **T0.09** | PASS | 2026-08-04 | `b30166a` — Colab round-trip, T1.02/T1.03/T1.07's GPU foundation |
| **T1.07** | PASS | 2026-08-05 | `55a07f4` "One job contract, actually: JACK_OUT and basename artifact keys" |
| **T2.02** | VOID | 2026-08-09 | 3 commits, incl. `d74e1bd` "T2.02's stale citation" |

These are the same shape as BA.02's flagged stale entry. BA.02 is flagged only
because it happens to be new enough to carry a stamp; these three are silent
for the opposite reason. The instrument's coverage is a function of *when a
spec last ran*, not of *whether it is at risk* — which is backwards, because
the specs that have not run recently are precisely the ones whose code has had
time to move.

I checked the other 27 too: all 30 sit on a `protocol.py` that has taken 23–26
commits since they ran. That is recorder hardening rather than measurement
change, and I am **not** calling it a finding — but it is the reason the honest
number here is 3 and not 30, and the reason a blob check beats a commit-count
heuristic. My own first pass over this said 8; seven of those were the
recording commit landing seconds after its own run.

**Why the planted positive did not catch it.** T0.17's control plants a
fabricated record with a moved dependency and proves the detector fires. It
does. What no test asserts is that **any real record is inside the detector's
domain** — a synthetic positive proves capability, not coverage.

**Fix (precise, in FOR THE BUILDER B1).** Make the check declaration-free.

---

## RANK 2 — MOST URGENT, and it is this document's fault: the owner is being steered toward an option GOAL.md forbids

**Zero damage to the ledger. It is first in urgency because it can cause an
irreversible architecture change on the owner's next one-line reply.**

`docs/DECISIONS_NEEDED.md:1216` — the **14th audit's own entry**, dated
2026-08-13 — says of D1 (*does the 57M trunk stay in the control path?*):

> *"**A — freeze the trunk for control, small dedicated policy head** — remains
> the loop's recommendation and the only option that explains the data…
> **One line settles it.** 'Do what the measurements say' will be read as A,
> journalled, and T2.01 re-run under the new architecture."*

`docs/DECISIONS_NEEDED.md:599` — an overseer entry from **2026-08-10, still
open, still unanswered, in the same file 585 lines earlier** — says:

> *"D1 — THE OPTION SET IS STALE: option A contradicts the PLASTIC-ONLY decree…
> A frozen 57M trunk with a small trained head is a frozen component inside
> Jack. **Option A is unconstitutional under the decree that postdates it.**"*

GOAL.md:76 is unambiguous: *"PLASTIC ONLY — nothing inside him is frozen
(owner decree, 2026-08-09). Every component inside Jack learns: his encoders,
his core, his fusion."*

Three compounding facts:

1. The 2026-08-10 entry asked for a fork the owner has never picked — **(i)
   strike option A**, or **(ii) keep A and narrow the decree in CHAMPIONS.md's
   scope paragraph**. Until (ii) is chosen, A is not on the menu.
2. The 14th audit's update **does not mention that entry exists**, and appends
   a pre-authorised trigger below it. The owner reading the file bottom-up —
   which is how an append-only file is read — sees the trigger and not the bar.
3. Item 4 of the 2026-08-10 entry named two artefacts still offering freezing
   as live. One is unfixed four days later: **`docs/CHAMPIONS.md:66`** still
   lists the Control-architecture seat's challengers as *"frozen-trunk+head vs
   tuned-PPO vs others"*.

That 2026-08-10 entry predicted this exact failure — *"the next agent to read
this file will design toward a recommendation the constitution forbids"* — and
the next agent that did so was the overseer. I am not re-litigating D1's
evidence, which is complete and unchanged. I am saying the **menu** is wrong,
the oversight organ propagated it, and a one-word owner reply could now enact
a barred change believing it was ratifying a measurement.

---

## RANK 3 — GPU receipts and the ledger cannot be joined by any field (carried, 3rd audit)

Compute honesty is currently checkable only by hand arithmetic on timestamps.

- **10 records were produced on remote GPUs. Exactly 1 (T1.02) carries a
  `gpu_job_id`.** The other 9 (T0.09, T1.07, T1.08, T1.09, T1.10, T2.01,
  T2.02, T2.03, T4.02) carry `metrics.gpu` / `metrics.backend` and nothing that
  names the job.
- `experiments/gpu_submissions.jsonl` records `attempt_id`, `job_id`, `head`,
  `pid`, `charge_seconds` — and **no spec id**. Neither side names the other.
- **All 9 report `hardware: "aarch64/Linux/torch2.8.0+cpu/cpu"`** — the
  dispatcher's box, not the P100 the work ran on. Nothing is fabricated (the
  truth is in `metrics.gpu`), but the field a reader trusts for *"where did
  this run"* says "cpu" for nine GPU results. Flagged by the 13th audit (B7)
  and the 14th (B5); unchanged.

**I did the join by hand, and it reconciles.** Both 5.58 h Kaggle jobs are
accounted for: `1786304547` (2026-08-09T19:42) → T2.01's FAIL recorded
2026-08-10T01:17, and `1786519461` (2026-08-12T07:24, 20 087 s) → T2.01's
current FAIL at 2026-08-12T12:59 (`duration_s` 20 097). No GPU hour this week
is unaccounted for. Worth naming plainly: **11.16 of W32's 18.65 Kaggle hours
— 60% — went to T2.01**, producing one honest FAIL on a spec whose
architectural question is parked on D1.

---

## RANK 4 — 11.23 Kaggle hours expire in ~47 h and the last three iterations did not touch them

`gpu_budget.json`, week 2026-W32 (Sunday-start, resets **Sun 2026-08-16**):

    kaggle productive  18.6496 h
    kaggle failed       0.1225 h   (2 jobs, correctly bucketed as waste)
    remaining          11.2279 h   of 30.0        overruns: []

The 14th audit's B3 was **discharged**: T4.02 and T3.07 were both implemented
and run (`03499a1`/`2d78651`, `741f7cf`/`0cfb066`) — the first entries ever in
T4-as-a-rule and T3. But T4.02's Kaggle job cost **0.117 h** and T3.07 ran on
CPU in 29 s, so B3 spent ~0.1 of the 11.35 h it was raised about. The journal's
queue then named **T2.04** (`gpu<20min`) for three iterations running
(22:07, 23:07, 00:07) and all three went to BA.02 instead — which was correct
work in itself (see §3) and has now ended in D8/park. With ~47 h and ~47
iterations left before reset, this is recoverable, not lost. It is the same
shape as last audit's finding and should not need a third one.

W31's 37.4554 h against a 30 h ceiling is **not** a live finding: `gpu.py:246`
and `:357` name it as the scar the `overruns` list was built from, and W32's
`overruns: []` at 18.65 h is correct.

---

## 1. Integrity of the ledger — NO FINDINGS beyond RANK 1

Checked all 79 PASS records:

| check | result |
|---|---|
| implementation exists in `experiments/tests/` | **79/79** |
| recorded `commit` still exists in git | **79/79** |
| spec declares a `control` | 77/79 (T0.01, T0.10 declare `None` — correct) |
| a declared control actually ran (`control_metrics` present) | **77/77** |
| **`_check` actually READS the control argument** (AST) | **77/77** |

The last row is the one worth stating: a control that runs and is never gated
on is the same as no control, and there are zero of those. `run_spec` also
refuses `control_fn` without a declaration (`UndeclaredControl`), and that
guard fired for real this week on T4.02 (`cdbb08e`) — a live mechanism, not a
decorative one.

---

## 2. Thresholds and controls over time — NO FINDINGS. Nothing loosened.

`git log -p --since="7 days ago"` over `registry.py`, `registry_expansion.py`,
`tests/` is **+23 949 / −208 across 69 files, 120 commits**. Exactly **four**
constant assignments were removed in the window, and every one moves the wrong
way for a cheat:

| removed | replaced by | direction |
|---|---|---|
| `MAX_VF_PG_RATIO = 50.0` | `MAX_VF_PG_GRAD_RATIO = 25.0` | tighter, on a re-registered quantity, diagnosed in the docstring |
| `TRAIN_MINUTES_PER_SEED = 22` | `110` | **5× more compute** |
| `N_ENVS = 8`, `ROLLOUT_STEPS = 128` | retuned upward | more compute |
| `SEED = 0` | `for SEED in (0,1,2)` | **1 seed → 3** |

Three lines matching `-control=` looked like deletions and are all re-adds with
strengthening (T2.01, T2.06, ME.7 — each gained `COVERS:` or notes, control
text byte-identical). `seeds` moved only upward (ME.5, ME.8: unset → 3). No
`_check` gained an `or`; T1.03 and T1.05 gained **conjunctions** (`and
c["planted_none_detected"] == 1.0`, `and control_moved`). Roughly 20 specs
gained a control they did not previously have.

**One judgement call, reported not as loosening but so it is on the record:**
T2.02's `depends_on` changed from `["T2.01"]` to `["T2.00","T1.08","T0.10"]`,
dropping a dependency on a FAILing spec, and its metric from
`return_at_2M_steps` to `return_at_matched_steps` (~640 K/seed). Both are
dependency/scope relaxations. Both are argued in the spec's own `notes` and
docstring with the measurement behind them (106 steps/s ⇒ 2M×3 seeds ≈ 16 h
against a 9 h session cap; the arbitration cannot be gated on its own subject
passing). Declared, dated, and reasoned — the opposite of silent. The
hypothesis and kill criterion are unchanged.

---

## 3. Drift from the goal — NO DRIFT. Every unit traces to a GOAL.md sentence.

| builder unit (last 24 h) | GOAL.md sentence it serves |
|---|---|
| BA.02 v2/v3/diagnosis (6 iterations) | *"proprioception & balance"* in the sensory inventory; balance is 1 of 17 zero-pass commitments |
| T3.07 ablate mood conditioning → FAIL | *"components that must EARN their parameters via ablation or be deleted"* |
| T4.02 fusion-boundary gradients → FAIL | *"a genuinely unified brain where every sense is load-bearing"* |
| B1 certificate re-runs (PG.1/2/4, T2.20) | *"protects the honesty of watching what happens"* |
| LC.03 registered run (in flight, ~9.5 h of 15–20 h) | *"learns his world by living in it"* |

Nothing on that list serves no sentence. Six of those iterations went to a
single spec that ended parked — but BA.02 ended by **measuring that its own
claim has no headroom in this body** (contrast ceiling ~0.0–0.1 s against a
pre-registered 0.20 s floor, four probes, 120 paired packs each), which is a
finding, not a failure to produce one. The alternative — quietly amending the
gate down to what the body can do — is the exact disease this project exists to
avoid, and the loop did not do it. Zero constants were moved across V2 and V3;
V3's single amendment (`BA02_TILT0_LOG10_DEG`) was chosen by a measured
headroom probe and registered before the run.

**The converse, which is the harder question.** Of the three claim families
GOAL.md most exposes to quiet neglect:

- **Curiosity** — 12 specs, **1 passing claim** (T2.08), and its apparatus PG.4
  was one of the four certificates that spent four days unverifiable. Eight of
  the twelve (CU.1–CU.7, T5.06, T5.08) have never been implemented.
- **All-senses fusion** — 21 specs under one-brain/unison, **1 passing claim**
  (UB.9). The rule spec that would show fusion is balanced, T4.02, now reads
  **FAIL** (ratios 29.8/12.5/29.1 against a pre-registered ≤10, plant control
  detected at ~12 000×, so the instrument sees). That is an honest and
  important red: the fusion is measurably lopsided, in a direction
  (touch-over-audio) nobody predicted.
- **Learning by living** — carried entirely by LC.03's in-flight run and by
  XL.00 (fixture). XL.01 *"death does not erase what he learned"* — the actual
  claim — is unimplemented.

---

## 4. Is the builder alive and productive? Yes, with one recovered outage.

24 h to 2026-08-14 00:37: **24 iterations, 21 rc=0, PASS 75 → 79 (+4)**, spec
count 166 → 169. Two detached multi-hour runs carried throughout without
disturbing them. Load 3.1–3.3 the whole evening (3 LC.03 workers at 101%),
never near the 6.0 abort ceiling; 12–13 GB free throughout.

**Three iterations lost** (10:07, 11:07, 12:07) to *"You've hit your session
limit · resets 1pm (UTC)"*, 3 s each — 12.5% of the day. The 14th audit's B4
asked for this to be countable; **it is implemented and I verified it**:
`scripts/ladder_loop.sh:173` writes `/data/jack-logs/lost_iterations.log` on a
session limit, tries the `opus sonnet` fallback chain first, and the next
successful iteration announces and clears the count. The file is currently
absent, which is correct — nothing has been lost since it shipped at ~15:00.

**On the flat stretch.** PASS has not moved since 14:20 — 11 iterations, ~10 h.
That is not stagnation: in that window the loop recorded T4.02 FAIL, T3.07 FAIL
and BA.02 VOID, raised D7 and D8 with full arithmetic, closed the 14th audit's
B1/B2/B3/B4, and appended four LESSONS entries. A day that converts three
optimistic hypotheses into three honest negatives is a good day on this ladder;
the counter just does not count it.

---

## 5. Compute honesty — accounted for. See RANK 3 and RANK 4.

Every W32 GPU hour reconciles to a ledger entry or to the `_failed` waste
bucket. The reconciliation required manual timestamp arithmetic because no
field joins the two records — that is RANK 3.

---

## 6. Stuck decisions

| item | age | read |
|---|---|---|
| **D1** — 57M trunk in the control path | **10 days** | Evidence complete since 2026-08-09. Blocks T2.01 (the ladder's oldest FAIL), T2.02 (VOID) and dependents. **Its option menu is broken — RANK 2.** |
| **D1 reconciliation** (option A vs PLASTIC-ONLY) | **4 days** | Raised 2026-08-10, never answered, and now actively harmful because a later entry recommends the barred option. |
| **D7** — MovementMoodCoupling failed its ablation | 1 day | Correctly the owner's: deleting a component is a Tier-3 law the loop may not execute alone. Evidence complete (0.225/0.275/0.375 vs chance 0.25; reference arm proves the map learnable at span 0.52). Not stuck yet. |
| **D8** — BA.02 unmeasurable in the rover body | <1 day | Correctly the owner's: a body change is a world-contract change. Recommendation (park) changes no certificate. Not stuck yet. |
| `/data` at 95%, Claude credits unmetered | standing | Unchanged; credits now have one measured instance (§4). |

**Could the system have decided any of these itself?** No. D7 and D8 both bottom
out in "delete a component" / "change the body", which SYSTEM.md reserves. D1 is
the one that could have been narrowed by the system — and it partly was: the
loop correctly *declined* to re-run T2.01 on 2026-08-13 (`a3b12f6`) on the
grounds that a re-run is a seed redraw against a 5σ bar, i.e. run-until-pass.
That is the right call and it deserves saying.

**Was any owner decision quietly acted on?** No. D2 (does a VOID dependency
block?) was resolved by the loop, but only after the 11th audit ruled it the
loop's to take, and it is recorded in full in `DECISIONS_RESOLVED.md` with
method, numbers, loser, and a re-open trigger.

---

## 7. Bakeoff hygiene — NO FINDINGS.

`DECISIONS_RESOLVED.md` holds three entries and all three are clean:

- **PS.01/J — VOID.** Every arm below the 3.0σ learning gate, so the bakeoff
  refused to arbitrate. A VOID recorded as a VOID, not as a verdict. Correct.
- **PS.01/J2 — WINNER `impl_speed`.** Clears the null by 10.32σ; beats
  runner-up `peak_dvel` by 2.66σ against `bakeoff.py`'s declared
  `margin_sigma = 1.5`. I checked whether 2.66σ is "inside the noise margin" —
  it is not: the margin is a pre-registered constant in the code, 2.66 > 1.5,
  and `bakeoff.py:268` would have returned TIE otherwise. The `screen` gate
  mode carries a written rationale (the arms are deterministic reductions of
  identical cached rollouts, so a low score cannot be a broken run).
- **D2 — WINNER BLOCK.** Decided by ledger replay rather than `run_bakeoff`,
  with the reason stated up front (the arms are two readings of a dependency
  graph, not learners — no seed noise, no null, nothing that could have
  failed). Pre-stated metric, both arms' numbers published, loser recorded,
  re-open trigger named. This is what the third law is supposed to look like.

---

## 8. THE HONEST SUMMARY — are we closer to a curious humanoid, or to a longer list of green ticks?

**Yesterday: closer to the humanoid, and the green ticks went nowhere. That is
the right trade and it should be said as praise.**

The ladder gained **zero** PASS in the last 10 hours and recorded **three
negatives** instead — and every one of them is about Jack rather than about the
harness:

- **T3.07 (FAIL):** mood does not reach behaviour in the shipped system.
  `style_net` and `posture_net` receive a gradient nowhere in the repo. A
  component that has been in the architecture diagram for weeks is measurably
  decorative.
- **T4.02 (FAIL):** the fusion is not balanced — 29.8× at the boundary against
  a pre-registered ≤10, and in the *unexpected* direction (touch over audio).
  "One brain, all senses in unison" now has a number attached, and the number
  is bad.
- **BA.02 (VOID → D8):** the rover body cannot catch itself in any envelope
  with any learner, measured four ways. A constitutional sense has a certified
  sensor (BA.01) and a claim that must wait for a body.

That is what the project said it would do. T3 and T4 were 0-and-1 for weeks
while T0/T1 sat at 41/41; the first two entries into the tiers that ask *"does
this earn its place"* and *"do the senses actually fuse"* both came back red,
and the loop recorded them, localised them, and escalated the parts it may not
decide. Compare the alternative history where BA.02's gate got quietly relaxed
to what the rover can do: three more green ticks and a ladder worth less.

**The counterweight, stated as plainly.** 17 of 23 constitutional commitments
still have nothing passing, unchanged for three audits. **T5 — "THE CLAIMS",
the tier that is the thesis — is 0 of 27 and has never had a single entry that
is not a VOID.** Curiosity, the north star, rests on one claim spec whose
apparatus was unverifiable four days ago. Nothing yet climbs anything. The
system is excellent at being honest about what it has not built, and that is
genuinely the harder half — but honesty about a gap is not the gap closing, and
the gap did not close yesterday.

And one thing got quietly worse rather than better: the two findings above are
both **audit organs failing at their own job** — a staleness detector that
reads green over a domain it cannot see, and an oversight report steering the
owner into a constitutional wall. The builder cannot catch either. That is the
whole reason this seat exists, and it means the seat's own output needs the
same suspicion as everything else.

---

# FOR THE BUILDER

**B1 (RANK 1, do this first).** Make the staleness check **declaration-free**,
because `IMPL_DEPS` cannot protect any record that predates it.

- In `run.stale_claims`, add a third kind for records lacking `impl_sha`:
  compare the implementation file's content **at HEAD** against its content
  **at the recorded `commit`**, falling back to the newest commit touching that
  file within 30 minutes of `ran_at` (the recording commit routinely lands
  seconds *after* its own run — comparing against the recorded commit alone
  reports 8 false positives where the truth is 3). If the blobs differ, report
  it as **stale**, not as "cannot be checked". This needs no declaration from
  anybody and covers all 30 records today.
- Today that fires on exactly three: **T0.09** (PASS), **T1.07** (PASS),
  **T2.02** (VOID). Re-run T0.09 and T1.07 — both are cheap — and let them pick
  up `impl_sha`. T2.02 is a VOID blocked on D1; leave it flagged.
- Then fix the *report*, which is the part that misled: `run stale`'s line
  *"(no declared dependency has moved)"* must also print **how many of those
  records declare a dependency at all**. Today that is `0 of 30`, and printing
  it would have made this finding self-evident. A detector that reports a count
  must report its denominator.
- Finally, add to T0.17 an assertion that the `UNVERIFIABLE_MOVED` domain is
  **non-empty over the real ledger**, or that the declaration-free check covers
  every unstamped record. T0.17's existing planted positive proves the detector
  *can* fire; nothing proves any real record is inside it.

**B2 (RANK 2's mechanical half — the judgement half is the owner's).** Correct
`docs/CHAMPIONS.md:66`. The Control-architecture (D1) seat still lists its
challengers as *"frozen-trunk+head vs tuned-PPO vs others"*. That is item 4 of
the still-open 2026-08-10 D1 reconciliation entry, unfixed for four days, and
`814ed89`'s plastic-only sweep missed it. Either strike `frozen-trunk+head` or
annotate it *"barred pending the D1 reconciliation (DECISIONS_NEEDED.md:599)"*
— do not silently leave a barred arm listed as live. Do **not** touch
`DECISIONS_NEEDED.md:73`/`:241`/`:1216`: those are owner-facing and the fork
between them is the owner's to pick.

**B3 (RANK 3, carried a third time — please close it this time).** Two fields,
one commit:
- Stamp `hardware` from the machine that **ran the work**, not the dispatcher.
  Nine GPU records say `aarch64/…/torch2.8.0+cpu/cpu`; the truth is already
  sitting in `metrics.gpu`.
- Make `gpu_job_id` mandatory on every GPU-dispatched record — have `submit()`
  return it and `run_spec` fold it in, rather than leaving each spec to
  remember (only T1.02 does). Symmetrically, write the spec id into
  `gpu_submissions.jsonl`'s attempt record. Today the receipt log and the
  ledger share no field, so *"which hours bought which result"* is answerable
  only by timestamp arithmetic — I did it by hand this audit and it reconciled,
  but that is not an audit trail, it is a coincidence of durations.
- This is the same designed unit as the 13th audit's live-receipt /
  charge-at-attempt item, still owed.

**B4 (RANK 4, time-boxed — 47 hours).** **11.23 Kaggle hours expire Sunday
2026-08-16.** T2.04 (`gpu<20min`) has been the named queue item for three
iterations and has not been implemented. Implement and dispatch it, and pick a
second `gpu<2h` spec behind it — GPU work runs remotely and competes with
neither LC.03 nor BA.02 for this box. If BA.02 is parked per D8, the standing
zero-pass rule's next pick is a commitment with a *runnable* claim spec:
**SM.02** (occluded food), **TA.02** (one-trial aversion) and **VO.02** are all
named runnable-today in `CHAMPIONS.md`, and all three sit on a certified
fixture.

---

# FOR THE OWNER

**1. D1's menu is broken, and my predecessor made it worse. Please read this
before you answer D1.**

The 14th audit's entry (`DECISIONS_NEEDED.md:1216`) told you *"'Do what the
measurements say' will be read as A"* — where **A is "freeze the 57M trunk,
small trained head does control"**. GOAL.md:76, your own decree of 2026-08-09,
says **"PLASTIC ONLY — nothing inside him is frozen… his encoders, his core,
his fusion."** A frozen trunk with a trained head is a frozen component inside
Jack. An overseer entry raised this on **2026-08-10** and is still unanswered
(`DECISIONS_NEEDED.md:599`); the 14th audit did not reference it.

So: **do not reply "do what the measurements say."** It would be read as A and
would enact something your constitution forbids.

What is actually needed is the one line that entry asked for four days ago:

- **(i) Strike option A.** D1 becomes B (split trunks) vs C (keep training
  end-to-end — reclassified from "refuted" to **untested**) vs D (delete the
  transformer from the control path).
- **(ii) Keep option A and narrow the decree** — say that PLASTIC-ONLY governs
  the *sensory* towers and a frozen *control* trunk is a separate question, and
  that scope goes into `CHAMPIONS.md`. This is defensible; the decree's stated
  reason is a sensory tower's reshaping gain.

The measurements themselves are not in dispute and have not changed: the 57M
trunk at 261/318 return against a 54 K MLP at 531 and a 125 K net at 530 —
failing a 3σ learning gate that a 125 K net clears at 7σ, across three
independent runs. **Ten days open. It blocks the locomotion branch and the two
GPU specs that could spend an expiring quota.**

**2. D7 — MovementMoodCoupling failed its ablation (new, 1 day, evidence
complete).** Mood's only route to behaviour scores 0.225/0.275/0.375 against
chance 0.25. A reference arm proves the map *is* learnable (span 0.52), so this
is a training/design failure, not an impossible component. Tier 3's law is
"dead weight is deleted", and deleting is yours: **delete (1,539 params;
T2.12's PASS that moods are separable is untouched)**, **redesign and re-run
T3.07**, or **keep it as declared cosmetics** — in which case no spec may cite
mood as a behavioural channel again. The loop recommends redesign-as-input-token
if mood is to be a sense at all, deletion if it is not.

**3. D8 — BA.02 cannot be measured in the rover body (new, <1 day).** The
sense is certified (BA.01 PASS); the *act* is not testable on a body with no
directional catch authority — measured ceiling ~0.0–0.1 s against a
pre-registered 0.20 s floor, four probes. The loop recommends **parking** it
until the humanoid body lands, which changes no certificate and no claim text.
This one needs a nod, not analysis.

**4. Nothing on the ledger is false.** Seventy-nine PASS records, all with live
commits and implementations, and every declared control both ran and was gated
on. No threshold loosened in seven days. The three findings above are about
auditability and about the decision record — not about a claim that is not
true.
