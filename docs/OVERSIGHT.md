# OVERSIGHT — 81st audit, 2026-09-07 06:37–07:0x UTC (at `4f1591d`, tree clean)

## VERDICT: INTEGRITY RISK — **`ME.11.A` holds a live PASS whose control can no longer execute at all, and the whole nine-module `ME.11` family certifies a retriever that was replaced underneath it yesterday. Neither `impl_sha` nor `T0.35` can see it, because the dependency is one hop away.**

The work in this window was good and I want that on the record before the
finding: the `ME.1` similarity-floor repair was decided by a real 5-arm × 3-seed
bakeoff with two rejected arms measured against named scars, not by argument;
`ME.9`'s aliveness floor was *tightened* 9 → 12; `T0.11` (the 34-day oldest live
certificate) and `T1.01` (with the new `mode_training` conjunct) were both
re-bought; every one of the Review's eight `FOR THE BUILDER` items is
discharged. Sections 2, 5, 6 and 7 are clean. **Nothing was loosened.**

The finding is that the *strengthening* broke something, silently, in a
direction no instrument in this repo watches.

On 2026-09-06 the Review strengthened `ME.1` (`35b9d51`, 06:52) and the builder
recalibrated `EpisodicMemory`'s floor (`6502d36`, 12:16). Both were correct. But
`me_1_event_log._build_life` gained a fourth return value, and
`EpisodicMemory`'s `abstain_below` default moved 0.34 → 0.95 with the similarity
metric itself replaced. Nine `ME.11` modules reach `EpisodicMemory.py`
**transitively** — through `me_1_event_log.py` or through
`fixtures/paraphrase_eval.py` — and not one of them declares it. `impl_sha`
hashes the test file plus its *declared, one-hop* deps. `T0.35`'s walker finds
undeclared *direct* imports. The transitive hop is covered by neither, so the
staleness lane printed a clean board over the entire family.

`ME.11.A` — *"Arm A, lexical containment, the incumbent, as the null"* — is the
worst case, and I verified it by execution rather than by reading:

```
$ python -c "from experiments.tests import me_11_a_lexical_incumbent as A; A._control(0)"
  File ".../me_11_a_lexical_incumbent.py", line 137, in _control
    mem, events, now = _build_life(seed, tmp)
ValueError: too many values to unpack (expected 3)
```

Its `_check` requires `c["templated_recall"] >= 0.80`. That control ran and read
0.85 on 2026-09-02. **It cannot be run today.** The audit's own standing rule is
*"a PASS whose control was never run is a claim without evidence"*; this is the
stronger case — a PASS whose control can no longer be run, held by the spec that
is the null for six other arms.

Ranked by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger — ONE BROKEN CONTROL, ONE STALENESS CLASS NOBODY WATCHES

The mechanical sweep is clean: **106 PASS rows, 0 dead commits, 0 missing
implementations.** Two PASS rows carry empty `control_metrics` (`T0.01`,
`T0.10`) — the same two argued exemptions the 80th audit recorded; unchanged and
not a finding.

### 1.1 `ME.11.A`'s control raises `ValueError` — CONFIRMED BY EXECUTION

| | |
|---|---|
| row | `ME.11.A` **PASS**, attempt 2, `c7325c2`, ran 2026-09-02T18:45:01 |
| recorded control | `templated_recall` 0.85 ± 0.014 (bar `MIN_TEMPLATED_RECALL` 0.80) |
| broken by | `35b9d51` (2026-09-06 06:52) — `_build_life` returns 4 values, was 3 |
| call site | `experiments/tests/me_11_a_lexical_incumbent.py:137` |
| today | `ValueError: too many values to unpack (expected 3)` |

`me1_floor_probe.py` (written the same day) was updated to the 4-tuple at lines
178 and 204. `me_11_a_lexical_incumbent.py:137` was not. It is the only other
consumer of `ME.1`'s `_build_life` in the tree, and nothing looked for it.

The commit that broke it says, accurately, *"strengthen-only — no bar moved, no
control weakened."* It weakened no control. It made a control in a **neighbouring
spec** uncallable, and the idiom that hides this — importing a sibling test
module's private helper — is invisible to every instrument here.

### 1.2 Nine `ME.11` certificates describe a retriever that no longer exists

`6502d36` replaced `EpisodicMemory`'s scorer: `abstain_below` 0.34 → 0.95, and
similarity changed from raw containment `|q∩e|/|q|` to **coverage over KNOWN cue
words** (`|q_known∩e|/|q_known|`, unknown cue words dropped). This is a different
retriever, not a tuned constant, and the builder said so plainly in the commit.

`fixtures/paraphrase_eval.py:433` builds the store as `EpisodicMemory(path=path)`
— **the bare default**. So every arm that reads through the fixture now reads
through the new scorer.

Transitive reach, computed statically over the tree (AST walk, relative imports
resolved), with `IMPL_DEPS` declarations checked:

| module | spec | status | hops to `EpisodicMemory.py` | declares it |
|---|---|---|---|---|
| `me_11_0_eval_set_honest.py` | ME.11.0 | **PASS** | 1 (via `paraphrase_eval`) | no |
| `me_11_a_lexical_incumbent.py` | ME.11.A | **PASS** | 1 (via `me_1_event_log`) | no |
| `me_11_b_bm25s_stemming.py` | ME.11.B | FAIL | 2 | no |
| `me_11_c_static_embeddings.py` | ME.11.C | FAIL | 2 | no |
| `me_11_d_minilm_onnx.py` | ME.11.D | FAIL | 3 | no |
| `me_11_e_weighted_hybrid.py` | ME.11.E | VOID | 3 | no |
| `me_11_f_cascade_rerank.py` | ME.11.F | VOID | 3 | no |
| `me_11_finds_from_paraphrase.py` | ME.11 | FAIL | 3 | no |
| `lg_00_not_a_puppet.py` | LG.00 | PASS | 1 | **YES** — correctly staled |

**8 modules, 0 declarations. 2 of them are live PASSes.**

Six of the eight *do* declare `experiments/fixtures/paraphrase_eval.py` — which
is the honest instinct and does nothing here, because a change to
`EpisodicMemory.py` does not change `paraphrase_eval.py`'s bytes. Declaring the
door does not hash what is behind it.

`ME.11.0`'s control is `leaky_null_recall >= 0.80`, recorded at **1.0** — the
aliveness proof that the paraphrase eval set is not impossible. It was measured
on the 0.34 retriever. `ME.11.A`'s headline is `paraphrase_recall_at_1 <= 0.10`
— the *null* against which arms B–F are judged, and the reason `ME.11` exists at
all. Both now describe a scorer that was deleted.

I am not claiming either verdict would flip. I am claiming **nobody knows**, and
that the board says otherwise.

### 1.3 The mechanism, stated generally — this is the lesson, not the incident

`protocol.py:impl_sha_of` hashes *the test file plus the files it declares*.
`protocol.py:undeclared_impl_imports` (78th audit B1, 2026-09-06) walks *the
module's own import nodes*. Both are **one hop**. A test module that reaches an
impl module through a fixture or a sibling test module is outside both, by
construction.

The `undeclared_impl_imports` docstring already tells this story in the
first person — *"when `EpisodicMemory.py`'s scorer was replaced that morning the
staleness lane printed a clean board over seven certificates"* — and the B2
repair (`80f8c80`) then enumerated *"the eight `EpisodicMemory` importers"* and
declared the set closed. It closed the direct set. The transitive set is eight
more modules and was never counted, on the same morning, by the same repair.

`T0.35`'s `GRANDFATHERED` allowlist is honest about what it exempts (`LF.01`,
`T2.10`, `T3.09` import directly and undeclared, and are named). The transitive
class has no allowlist because it has no detector.

---

## 2. Thresholds and controls over seven days — CLEAN, and I checked the risky ones myself

`git log -p --since="7 days ago" -- experiments/registry.py
experiments/registry_expansion.py experiments/tests/`. Every numeric movement in
the window is in the **tightening** direction:

- `me_9_attributed_recall.py`: `MIN_DISTRACTOR_EVAL` **9 → 12** (aliveness floor
  raised).
- `ME.3`/`ME.5`/`ME.9`/`ME.10` each **gained** `distractor_abstention >= 0.95`
  plus an evaluation floor as required conjuncts (`80f8c80`), on their own
  stores. `ME.3`'s `_check` gained two `and` clauses, not an `or`.
- `ME.1` gained `distractor_evaluated >= 30 and distractor_abstention >= 0.95`
  (`35b9d51`) at its own unchanged 0.95 bar.
- `T1.01` gained `mode_training` and a frozen `improvement_ratio < 1.5`
  conjunct; the frozen arm measures 1.00, so ~50% headroom, measured before
  being asserted.
- `W1.00`/`W1.02` register with `SIGMA_GATE = 3.0` **imported** from `W0.DIAG`
  rather than retyped, and `W1.02` reads `DP.04`'s `MIN_GAIN` live from source
  text with a `V1 uncalibrated borrow` VOID if it cannot.

No seed count fell. No control was deleted or weakened. No assertion was
removed. **No finding.**

### 2.1 The one change I want to name as *correct*, because it looked like the disease

`EpisodicMemory`'s `abstain_below` 0.34 → 0.95 reads like a loosened constant
and is the opposite. The metric it governs was replaced in the same commit
(coverage-over-known-words, not raw containment), so the two numbers are not
comparable, and the choice was made by `experiments/tests/me1_floor_probe.py` —
5 arms × 3 seeds, with the two cheap one-constant repairs **measured and
rejected** against named scars (raw@0.60 kills verbose recall to 0.000 on all
seeds; margin@0.20 kills terse attribution to 0.000). `ME.1`'s own 0.95 bar and
exclusion filter were untouched, `cued_recall` held at 0.85. That is law 3
working. The docstring even names the hole the new floor opens rather than
hiding it. I checked this one hard because it is exactly the shape of a silent
loosening, and it is not one.

### 2.2 An observation on `ME.1`'s new conjunct, for the Review — not a violation

Under the coverage floor with a 3-content-word cue, `recall` returns only events
containing **all three** cue words. `ME.1`'s distractor control excludes any cue
where a retained event carries all three (`me_1_event_log.py:151`). Those two
predicates are now the same predicate, so `distractor_abstention` reads 1.0000
by construction for this scorer family — it will still fire if the floor drops
(that is what caught 0.0000 yesterday), but it can no longer distinguish
principled abstention from the floor's arithmetic. This is a note for whoever
next re-reads the control, not a finding: the conjunct was honest when written
and is doing real regression work.

---

## 3. Drift from the goal — NONE in the window; the standing hole is unchanged

Last 24 h of builder work, each traced: `LEARNING_CORE.md` parameter-table
reconciliation (serves *"complexity must earn its place"* — the table drove
`LC.06`'s seated-arm concession, corrected 83% → 14.9%); three `protocol.py`-
staled certificate re-buys (serves *"the ledger is the only scoreboard"*);
`T0.11`/`T1.10` harvest; `LC.07` CPU-venue pricing. **Nothing serves no
sentence.** The builder also *declined* three units with reasons and corrected
two wrong next-slot pointers rather than executing them — that is the behaviour
this desk wants and it is worth saying so.

The converse, which is the harder question, is unchanged and I am repeating it
rather than re-deriving it: `coverage` reports **4 CLAIM-DEAD commitments**
(smell, balance, shelter/building, thermal-kills) where every claim spec is
parked or foreclosed, and **9 more with live claim specs and nothing passing**
(touch, tool use, told world, proprioception, death & retry, plasticity, sleep,
hunger/thirst, fast/slow). `coverage` exits 2 on exactly this. Curiosity: 12
specs, 2 pass. One brain / unison: 25 specs, 1 pass.

The Review's own completeness audit added the sharper version yesterday —
`GENERALITY.md` names 14 barriers, 4 have specs, all 4 `NOT_RUN`, **0 have a
passing spec** — and I concur with its framing that no instrument here will ever
raise it, because each measures the ladder we built.

---

## 4. Is the builder alive and productive? — ALIVE, HONEST, AND STARVED

24 iterations in the last 24 h, **24 of 24 `rc=0`**, zero paused, zero aborted
on load, `lost_iterations.log` still 0 bytes. Meters read and named every slot;
`week:all models` reset 04:59 and reads 1%.

**PASS delta over 24 h: 105 → 106 (+1). The last 17 consecutive iterations moved
it 0.** That is not idling — I checked the journals against the commits and each
slot did real, named work (certificate re-buys, a parameter-table
reconciliation, three corrections of wrong instructions). It is the board being
thin: `run next` offers 12 specs and every one is settled, held, parked or
foreclosed, and the four claim-dead commitments all wait on redesigns owned by
the Review. The builder said this itself — *"the constraint is upstream on the
Review's redesigns, not on builder uptime"* — and the evidence agrees.

The honest reading: **the builder is now rate-limited by the Review's design
queue, and the Review's design queue is over-subscribed** (see §7).

---

## 5. Compute honesty — 17.61 GPU-h bought a second consecutive VOID

GPU weeks charged: W32 16.61 h · W33 7.89 h · W34 1.62 h · W35 19.20 h ·
**W36 17.73 h of 30 — 12.27 h remaining, resets Sunday.**

Essentially all of W36 is one job: **`D1.0` attempt 2, 17.61 GPU-h across four
kernels, harvested VOID.** Attempt 1 (W35) was also VOID. So `D1.0` has consumed
**~35 GPU-h across two dispatches for two VOIDs** — and it is the repair path for
`T2.01`, which tops the frontier at frees 34 / blocks 38.

This is not waste and not dishonesty: both VOIDs are the learning gate firing on
the **denominator**, not on the arms (attempt 1: `c_e2e` 2.56σ vs a 3.0 bar;
attempt 2: the adopted gate fired on the *untrained twins* at 3.95σ/3.91σ while
every trained arm cleared 10.5–13σ). Both were committed as found, neither was
re-rolled, and the gate that produced the second VOID was committed **44 seconds
before** dispatch — I verified that ordering in the 80th audit. The arithmetic is
now on `d10-successor-rerun-under-adopted-gate` for the Review's 09-08 sitting.

The number worth carrying: the twins' means are identical across both attempts
(198.4/197.6) and only the random denominator's spread moved (30.27 → 22.12). A
third dispatch at ~17.6 h against a 12.27 h remaining balance is not affordable
this week, and a third VOID on the same denominator would be the third draw at
one outcome — the shape `T2.11`'s park exists to refuse. **No third dispatch
without a change to the denominator.**

---

## 6. Stuck decisions — CLEAN

`decisions --check` **EXIT 0**, ratchet ok: 0/10 undeclared, 0/3 unrouted
owner-asks, 0/0 vanished, 0/0 expired defaults. **No `MEANS-ESCALATED`.** No
`OVERDUE`. Nothing on the owner's desk that a measurement could settle.

`D24` (Learning-core venue, `class: goal`, `decide_by` **2026-09-11**) is armed
with default (iii) DECLARE-DO-NOT-DECIDE, and I agree with the Review that it is
the only legal default of the three — (i) commits four months of the entire GPU
budget by silence, (ii) is a threshold move by silence. `PROGRESS #1` is
correctly attributed to it.

No owner decision was acted on without being recorded. `D16` fired 09-06 by
armed default and the Review closed `t027-preserved-failimpl-as-artifact` on it
this morning, option (b), with `T0.27` left RED — a visible failure preferred
over a manufactured green, which is the right call.

---

## 7. Bakeoff hygiene — CLEAN, and one structural warning

`champions --check` **EXIT 0**, ratchet ok. No decision in `DECISIONS_RESOLVED.md`
was made without a learning gate; no VOID was treated as a verdict — the two
`D1.0` VOIDs were recorded as VOIDs and blocked the dependent decision rather
than resolving it, which is the behaviour the rule exists to produce; no winner
was chosen inside a noise margin.

Standing, unchanged, and not re-litigated here: 2/3 unfalsifiable seats (ASR,
Speaker ID), 2/2 unverified verdicts (Learning core `LC.03=VOID`, World *"no
deciding run named"*), 3/3 trigger debt. The World seat is still held BY VERDICT
— the file's strongest marking — with no deciding run.

**The warning is `review-queue`, which exits 0 and should still be read.** It is
0 violations, but the DUE-DATE PILE is amber on five dates and **10 live rows
share 2026-09-13 against a measured consumer capacity of 1 dated row per
cycle.** 25 live rows were dated onto a day that already carried its capacity.
The instrument correctly calls this a metric and not a violation — each may have
had a good reason — but §4 found the builder starved on exactly this queue. Ten
promises scheduled to break together on one Sunday, feeding a builder with
nothing else to do, is the shape of next week's finding. `review-queue` names the
mechanical answer: **2026-09-17 is the next date carrying no promise.**

---

## 8. The honest summary — closer, and the ladder we are climbing got shorter by one rung we did not notice

We are closer, and I can name the step: yesterday this project discovered that
the memory `GOAL.md` says makes him *him* invents an answer on 100% of the
questions it should refuse, and then **fixed it by measurement** — five arms,
three seeds, two cheap repairs rejected against scars the codebase had already
paid for, `cued_recall` held at 0.85. That is the ladder-and-apple standard
applied to our own instruments, and the demonstrated count went *down* before it
went up. Real.

What worries me is the shape of what I found. The strengthening was correct, the
repair was correct, and between them they left a live PASS whose control raises
`ValueError` and eight certificates over a retriever that no longer exists — and
**every organ reported green.** `status` was clean. `coverage`, `decisions`,
`champions`, `review-queue` all exited 0. The staleness lane, which exists for
precisely this, printed nothing, because the dependency was one hop away and
every detector we own is one hop deep. This is the missing-spec disease in a new
place: not a claim nobody wrote, but a *dependency edge* nobody can name, and
the same property makes it invisible to everything.

So: closer to a curious humanoid on the memory axis, genuinely. But the ladder
of green ticks now has a rung that is measuring nothing, and it got there
through two commits that were both, individually, the right thing to do. The
count that matters is not 106. It is that 106 includes at least one row that
cannot be re-bought without a code change nobody has made, and we found that by
running a control by hand rather than by any instrument telling us to.

---

## FOR THE BUILDER

Ranked. 1 is small and time-critical; 2 is the durable repair.

1. **Fix `me_11_a_lexical_incumbent.py:137` and re-buy `ME.11.A` and `ME.11.0`.**
   The line is `mem, events, now = _build_life(seed, tmp)`; `_build_life` now
   returns four values (`me1_floor_probe.py:178` shows the idiom:
   `mem, events, now, _ = _build_life(seed, tmp)`). This is a **one-token fix to
   a call site, not a spec change** — no threshold, no control, no bar. Then
   re-run both specs and **commit the rows as the runner writes them.**
   `ME.11.A`'s headline (`paraphrase_recall_at_1 <= 0.10`) and `ME.11.0`'s
   control (`leaky_null_recall >= 0.80`) were both measured on the 0.34
   retriever and may now read differently. **If either FAILs, that is a real
   finding and must be committed as one** — `ME.11.A` is the null six arms are
   judged against, so a changed null is a fact the whole family needs, not a
   re-roll. Do not "fix" it by pinning `abstain_below=0.34` in the fixture to
   restore the old numbers; if the incumbent must be frozen at 0.34 for the
   bakeoff to remain comparable, that is an argued spec amendment with the
   reason on the row, not a quiet constant.

2. **Close the transitive-staleness hole — `impl_sha` must follow intra-repo
   imports.** Today `impl_sha_of` hashes the test file plus its declared,
   one-hop deps, and `undeclared_impl_imports` walks one module's own imports.
   Eight `ME.11` modules reach `EpisodicMemory.py` at 1–3 hops and none declares
   it; `paraphrase_eval.py` is declared by six of them and hashing it catches
   nothing, because the impl module is behind it. The repair is in
   `protocol.py`, beside the two functions, in their idiom: **resolve
   `experiments.*` imports transitively and fold the reached repo-root modules
   into the hash** (or, if that changes too many recorded shas at once, add a
   `transitive_impl_imports` predicate and a `T0.35` property that FAILs on an
   undeclared transitive reach, with the current eight named in `GRANDFATHERED`
   so the floor follows the number down under P4's existing rule). Whichever
   shape: ship it **with a mutation falsifier** — touch `EpisodicMemory.py`,
   assert `ME.11.A` goes stale. A staleness detector that cannot be shown to
   fire is the thing this finding is about.

3. **Add the sibling-helper edge to that same walker, or stop the idiom.**
   `ME.11.A` broke because it imports `_build_life` — a private helper — from
   another spec's test module, and `ME.1`'s author had no way to know. Six other
   modules do the same across the `ME.11` chain, and `t2_10` declares
   `experiments/tests/me_11_a_lexical_incumbent.py` in `IMPL_DEPS`, which is the
   right instinct and the only place in the tree that has it. Either make that
   declaration mandatory for cross-test imports (checkable in the same walker)
   or move shared harness code into `experiments/fixtures/`. Naming which you
   chose is enough; I am not prescribing the shape.

4. **Do not dispatch `D1.0` attempt 3 this week.** W36 has 12.27 h of 30 left
   and the run costs ~17.6 h — it does not fit, and a third dispatch against an
   unchanged denominator would be the third draw at one outcome. The
   twin-denominator arithmetic is already on
   `d10-successor-rerun-under-adopted-gate` for the Review's 09-08 sitting;
   that row decides, not a re-roll.

5. **`audit_supersedes_fail` prints "that implementation was never committed"
   for `LG.00` and `T0.29` when both have hash-verified bytes under
   `refs/jack/failimpl/`.** Routed to you by the Review's own 09-07 DAILY close
   on `t027-preserved-failimpl-as-artifact`; I am repeating it here because it
   is a truthfulness defect in a standing-red instrument and standing-red
   instruments are where false text survives longest.

---

## FOR THE OWNER

1. **NO-DECISION — a report, and nothing here needs your ruling.** One passing
   certificate in the ladder (`ME.11.A`) cannot currently be re-bought: its
   control raises `ValueError` because a helper it borrows from a neighbouring
   spec changed shape yesterday. Eight specs in the same family certify a memory
   retriever that was replaced yesterday afternoon and none of them was marked
   stale, because our staleness check only looks one dependency deep. The repair
   is a one-line call-site fix plus a detector, both routed to the builder above,
   and neither needs you. You are seeing it because the count on your scoreboard
   said 106 all night and one of those 106 was not standing up.

   The context that makes it worth your minute rather than mine: this was caused
   by **two commits that were each individually right** — the Review honestly
   strengthening a control that had been passing vacuously for 29 days, and the
   builder honestly repairing the module underneath it by bakeoff. Good work in
   two organs composed into a hole in a third. That is the failure mode this
   project should expect more of as the ladder gets denser, and the durable
   answer is item 2 in FOR THE BUILDER, not more care.

2. **NO-DECISION — a rate report you may want to act on later, flagged now so it
   is not a surprise.** The builder is healthy (24/24 iterations `rc=0` in 24 h)
   and moved the demonstrated count **+1 in 24 hours**, with the last 17
   iterations at zero. It is not idling; every slot did named work. It is
   starved: all 12 specs the board offers are settled, held, parked or
   foreclosed, and the four claim-dead commitments — **smell, balance,
   shelter-building, and "too cold kills him"**, three of which you named
   constitutional on 2026-08-09 — all wait on world redesigns owned by the
   Review's queue. That queue has **10 rows promised on 2026-09-13 against a
   measured capacity of 1 per cycle.**

   I am not asking you to rule. The instruments are all green and correctly so.
   But the honest reading of §4 + §7 together is that this project's throughput
   is now set by one desk's design capacity, not by compute, credits, or the
   builder — and if that stays true through 09-13, the visible symptom will be
   ten broken promises in one day rather than a slow queue.

3. **Standing, re-stated not re-routed: `D24` decides 2026-09-11** (Learning-core
   venue, ~526 GPU-wall-hours against 30 h/week). Default (iii) —
   declare `VENUE-UNAFFORDABLE`, change nothing else — fires on silence, and I
   agree with the Review's recommendation and its reasoning: (i) commits four
   months of the entire GPU budget by silence and (ii) is a threshold move by
   silence. The price under (iii), stated plainly as it was on the row: this
   project cannot currently contest its own learning-core choice, and that stays
   true until the budget or the venue changes.
