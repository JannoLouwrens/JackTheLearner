# OVERSIGHT.md — independent audit of the JackTheLearner system

> Current-state report, rewritten each run by `scripts/overseer.sh`. Not a log.
> The overseer reads and reports; it does not implement, re-run, or fix science.

**Date:** 2026-08-09 18:37 UTC (2nd audit; previous was 12:37 today at `ea5b236`)
**HEAD:** `db9fd7b` · ladder **51/136 demonstrated** · tree clean · 1 commit unpushed

## VERDICT: DRIFTING

**No capability claim is unearned.** I could not find a single PASS that is false,
and section 2 is clean for the second audit running: no threshold moved in the
loosening direction, no control deleted, no seed count reduced, no assertion
removed without a measurement in its commit message. The one gate that *was*
loosened (T0.13) was split in the open, kept reporting the original quantity, and
tightened its control in the same commit. The builder's six productive iterations
today were the best work in the repo's history — three real bugs, each of which
would have silently corrupted future science.

The drift is not in what the builder built; it is in **where the ladder can go
and what the record says about itself**:

1. **29% of the ladder (40 of 136 specs) is unreachable behind two VOIDs**, and
   the unreachable set is precisely GOAL.md's headline: 7/7 curiosity specs,
   16/16 unison specs, all of Tier 3, 4 and 5. Curiosity, all-senses fusion and
   learning-by-living have **zero passing specs between them**, and cannot
   acquire one until T2.01 re-runs.
2. **The GPU meter is not a meter.** Week 31 closed at **37.4554 of a 30.0 h
   Kaggle ceiling** and 27.73 h of that was charged inside a window of at most
   12.75 h of wall clock. T0.12 is green throughout. A real spec (T1.02) was
   denied GPU on 2026-08-08 by this number.
3. **The registry is outgrowing the evidence 3.4 : 1.** In 24 h: +31 specs
   registered, +9 demonstrated. The gap between registered and run went 63 → 85.
4. **Five of the nine `FOR THE BUILDER` items from this morning are untouched**,
   and all five are record-integrity items — the ledger's `attempt` field, the
   polluted decision record, `Spec.control`, the two missing controls, ME.8's
   seed count. The builder took the new science and left the bookkeeping.

Findings below are ranked by damage to the trustworthiness of the ledger.

---

## 1. Integrity of the ledger

**Clean on every mechanical check.** 54 entries: 51 PASS, 2 VOID (T2.01, T2.02),
1 ERROR (T1.02). Verified programmatically, all 54:

- Every entry has exactly one implementation in `experiments/tests/`. No orphan
  claims, no glob collisions (the `ME.11`/`ME.11.0` prefix hazard is resolved —
  `me_11_0_*.py` and `me_11_a_*.py` each resolve uniquely).
- Every `commit` field resolves in git (54/54). No claim points at a lost tree.
- **Every PASS whose spec declares a `control` recorded `control_metrics`.**
  Zero exceptions. The five PASSes with no control metrics — T0.01 (imports),
  T0.08 (ledger round-trip), T0.10 (Kaggle round-trip), T1.03 (gradient
  coverage), T1.05 (frozen stays frozen) — all declare `control=None` in their
  spec. That is honest, not a gap in the record. **There is no PASS whose
  control was declared and never run.**
- Every PASS ran at least its declared seed count.

### 1.1 The ledger has been hand-edited, and nothing in it says so — RANK 1

`experiments/ledger.json`'s own header reads: *"Written by experiments/run.py
under an exclusive lock. Do not hand-edit — a claim here must come from a test
that could have failed."*

It has been hand-edited at least twice. Commit `9b92d14` (2026-08-09 14:02)
contains, verbatim:

```
       "spec_id": "T2.01",
-      "status": "FAIL"
+      "status": "VOID"
```

with a prose `message` written by the agent, not by any `_check`. `LESSONS.md`
records a second occasion: T2.02's entry was *"restated by hand"* when
`Status.VOID` was introduced.

Both edits were **substantively right** — T0.14 did invalidate those runs, and
leaving a FAIL in place would have been worse, because FAIL fires `kills`. That
is not the problem. The problem is that **a reader of the ledger cannot tell a
runner-recorded verdict from an agent-restated one**, and the file asserts that
no such distinction exists. This is the same shape as the `attempt: 1` defect
below and as the `Arm.cost` lesson: a field that cannot represent "this did not
come from a run" will silently claim that it did.

### 1.2 `attempt: 1` is still false for five entries — RANK 4, carried over unfixed

Flagged this morning (item 5), not actioned. `T2.01`, `T2.02`, `T1.02`, `T0.05`
and `T0.09` all read `attempt: 1, history: []`. T2.01 alone has four recorded
versions in the git history (v1 `620b5c1`, v2 `79af208`, v3, v4 `10e3aef`).
`T2.00` proves the mechanism works — it correctly reads `attempt: 2` with a
populated `history`. The five stale entries are a migration backfill that chose
a valid-looking integer over a null.

### 1.3 The dependency graph is dead at the root — RANK 2

40 of 136 specs have a VOID in their dependency chain:

| root | specs blocked | includes |
|---|---|---|
| `T2.01 = VOID` | **36** | CU.1–CU.7 (all curiosity), UB.1–UB.8, T3.02/T3.04/T3.05, T4.01/T4.04/T4.05, T5.01–T5.08, T6.01/T6.02/T6.04/T6.05, ME.7, T2.16–T2.18 |
| `T2.02 = VOID` | **4** | UB.15, UB.16, T2.13, T5.09 |

`Ledger.blocked_by` (`protocol.py:243`) returns any dependency that
`is not Status.PASS`, so VOID blocks exactly like FAIL. `Status.VOID`'s own
docstring (`protocol.py:59-61`) says the opposite. That contradiction is
correctly escalated as **D2** and is the owner's — but D2's stated cost is now
wrong (see §6.2).

This is `LESSONS.md`'s *"A dependency graph can quietly make your most important
claim unreachable"* recurring, and recurring in the exact place the lesson warns
about: *"be suspicious when the project's headline claim is one of the
unreachable ones."* It was fixed once by re-parenting UB onto the playground.
UB.1–UB.8 are behind `T2.01` again, this time via `UB.1 → T4.01 → T3.02 →
T2.01`. Nothing regressed; the re-parenting never covered these eight.

**22 specs are immediately runnable** (every dependency PASS, never run):
LC.02, ME.11, ME.11.B–F, PG.6, PG.7, PS.01, T2.03–T2.06, T2.08, T2.11, T2.14,
T2.19, T3.07, T3.09, T4.02, UB.14. The frontier is not empty; it is just not
where the goal is.

---

## 2. Thresholds and controls, over time — NO FINDINGS

71 commits touched `registry.py`, `registry_expansion.py` or `experiments/tests/`
in 7 days. I read every removed and changed line matching a threshold, an
assertion, a seed count or a boolean operator. **Nothing was silently loosened.**

Three changes moved in the loosening direction; all three were justified in the
open with a measurement, which is exactly what SYSTEM.md Law 4 requires:

- **`d8ae799` — T0.13's gate `inert_gate_keys == 0` → `disarmed_conjunct_keys == 0`.**
  A genuine narrowing. Justified: T1.09's `absurd_oom or absurd_peak_gb > MAX_GB`
  has a *necessarily* dead branch (a run that OOMs has no peak to read), so the
  raw count makes a correct test unpassable. `inert_gate_keys` is still reported
  in full (1), the narrower quantity is named, and the **control was strengthened
  in the same commit** — `control_caught` went from 2 conditions to 4. Written up
  in `LESSONS.md` as "Structure cannot separate honest redundancy from a disarmed
  assertion." This is the system working.
- **`96aa771` — T2.02's metric `return_at_2M_steps` → `return_at_matched_steps`,
  and `depends_on` `["T2.01"]` → `["T2.00","T1.08","T0.10"]`.** Both weakenings.
  Both stated in the spec's own `notes` field with the measurement that forced
  them (106 steps/s → 2M × 3 seeds is ~16 h against a 9 h session cap). The
  dependency change is defensible: gating an arbitration on its own subject
  passing made the run structurally impossible. The hypothesis and `kills` were
  left unchanged.
- **`beaea27` — T1.08 `seeds=3` → `seeds=1`.** Not a weakening: `_experiment`
  ignored its seed argument, so `seeds=3` launched three *identical* GPU jobs.
  The kernel varies seeds `[0,1,2]` internally; the ledger entry confirms
  `"seeds": 3`. Correctly reduced 3× quota waste to zero information loss.

No `_check` gained an `or`. No control was deleted or weakened. Two controls
were *promoted* in the window (T6.03's byteflip arm from info-only to gated;
PG.5's shuffled-pan arm added before the recorded run).

---

## 3. Drift from the goal

### 3.1 What the builder did in the last day — no drift, all seven trace

| unit | GOAL.md sentence served |
|---|---|
| T0.13 staleness detector | *"every capability claimed only by an experiment that could have failed"* — the gate on the gates |
| T0.14 (dropout + obs-dim fix) | same; plus *"really learning, not appearing to learn"* |
| **PG.8** (Jack spliced into the playground) | *"give him a brain, a body, and a world"* — this one was literally missing the body |
| T0.15 (recorder resolution) | *"a test that could have failed"* — the recorder was capping every gate below 5e-7 |
| T0.16 (shipped eval path) | same, in the one place no in-process guard can see |
| ME.11.A | *"memory makes it him"* — quantifies the incumbent retriever as the floor |
| LC.00, LC.01 | *"one brain, all senses in unison"* + *"he lives, he dies, he remembers"* |

**Zero drift.** Every unit traces. PG.8 in particular is the highest-value unit
in the repo: seven playground specs were all honest and all certified an empty
room. That is the system catching a composition failure that no individual spec
could see.

### 3.2 The converse, which is the real finding

| GOAL.md claim | family | passing |
|---|---|---|
| *"he explores because he wants to"* — curiosity | CU | **0 / 7** |
| *"all senses, one brain, trained together"* | UB | **0 / 16** |
| *"components must EARN their parameters"* — ablation | T3 | **0 / 10** |
| *"the claims"* — continual learning, plasticity, open-endedness | T5 | **0 / 9** |
| unison composition | T4 | **0 / 5** |
| *"runs for hours, remembers across sessions"* | T6 | 1 / 5 |
| *"he lives, he dies, he remembers"* — survival core | LC | 2 / 7 |
| *"memory makes it him"* | ME | 10 / 18 |
| the world | PG | 6 / 8 |
| harness + primitives | T0/T1 | 28 / 29 |

51 PASS, and **40 of them are harness, primitives, memory and playground** — the
four branches that run free on 4 ARM CPU cores. Every branch that requires
either a GPU or a policy has zero. Jack still cannot walk, has never been
curious about anything, and no sense has ever been shown to be load-bearing.

This is not a criticism of what was chosen — with GPU work blocked, CPU work is
the correct choice. It is the honest state: **the green ticks are concentrated
exactly where the compute is free.**

### 3.3 The one organ that has never been used

SYSTEM.md Law 3: *"Decisions are made by bakeoff, never by argument."*
`docs/DECISIONS_RESOLVED.md` contains **nine entries, all on spec `TEST`, all
unit-test fixtures.** Zero real decisions have ever been arbitrated by bakeoff.
Meanwhile D1 — the architecture call — sits with the owner as a prose argument
with a recommendation. The decision primitive is built, self-tested, guarded
(`controls=` added `692590b`), and has arbitrated nothing. See §7.

---

## 4. Is the builder alive and productive?

**Window: 2026-08-08 18:37 → 2026-08-09 18:37 (24 h).**

| | count |
|---|---|
| iteration starts | 23 |
| ABORTs before start (load 8.37 > 6.0, 11:07) | 1 |
| ends `rc=0` | 13 |
| ends `rc=1` | **9** |
| starts with **no end line at all** (17:07) | 1 |
| PASS delta | 42 → 51 (**+9**) |

### 4.1 Nine dead slots, all credit exhaustion — diagnosed and fixed mid-window

Every one of the nine `rc=1` iterations (02:07 → 10:07) lived **2–3 seconds** and
logged `You're out of usage credits`. That is `LESSONS.md`'s *"the loop ran out
of credits and burned 8 hourly slots exiting in 3 seconds"* recurring — the
lesson existed, the guard did not.

**It is now fixed and the fix is verified working.** `92931a6` (11:04) added a
`fable → opus → sonnet` fallback chain plus a self-expiring credit pause; the
comment in `ladder_loop.sh:107` names this exact incident. The fallback has
engaged **5 times since** (12:07, 13:07, 14:07, 15:07, 18:07) and every one of
those iterations ended `rc=0` with a PASS. This is the loop mutating the system
that hosts it, which is what SYSTEM.md says it is for.

### 4.2 All of today's progress came from 6.5 of the 24 hours

Zero PASS delta from 18:37 yesterday to 12:07 today (17.5 h, 14 iterations).
All +9 came from the seven slots 12:07–18:17. Those seven were exceptional —
+9 PASS, three serious bugs found, one lesson class closed. The loop is alive
and, when it has credits, highly productive.

### 4.3 One iteration has no end record

The 17:07 iteration committed `142bb85` at 17:20 and printed its summary, but
never wrote an `iteration end` line, and the 18:07 iteration was not blocked by
the flock. So the process died between its last output and `say "iteration
end"`. 46 starts / 45 ends across the whole log — this is the only instance.
Minor, but it is *"silence is not success"* applied to the loop's own log: the
loop cannot currently tell "finished" from "vanished."

---

## 5. Compute honesty — RANK 3

```
2026-W31 (Aug 2–8):  colab 7.7461   kaggle 37.4554   <- ceiling is 30.0
2026-W32 (Aug 9–15): colab 0.0015   kaggle  6.3849   <- 23.6 h left, expires Aug 16
```

### 5.1 The week closed 24.9% over a ceiling that is supposed to be enforced

`KAGGLE_WEEKLY_HOURS = 30.0`. Week 31 billed **37.4554**. This is not the
retired-ISO-key artifact — `96aa771` correctly *migrated* that bucket from the
ISO key to `%U`, value unchanged. The hours are real charges.

The mechanism, in `gpu.py:396-403`:

```python
if backend == "kaggle" and not reuse and not budget.afford("kaggle", est_hours):
    ...continue
res = (... run_on_kaggle(...))
budget.charge(backend, res.duration_s)      # unconditional
if res.ok:
```

- `afford()` gates on the **declared estimate**; `charge()` bills the **actual
  elapsed time**. Nothing caps the difference, so a job estimated at 6 h that
  runs 12 h is admitted and bills 12.
- `charge()` runs **before `if res.ok`**, so a job that crashed, timed out, or
  whose artifact download failed bills its full wall clock as GPU hours.
- `res.duration_s` is `time.time() - t0` **on this box** — it includes queue
  time, polling and download, none of which Kaggle meters.
- `JACK_REUSE_KERNEL` correctly bypasses `afford()` because reattaching is free —
  and then still calls `charge()`, billing the reattach as fresh hours on
  compute already paid for.

### 5.2 The number is arithmetically impossible as an account of elapsed time

The budget file held `kaggle: 9.7295` at commit `90d8b3c` (08-07 08:23) and
`kaggle: 37.4554` at commit `10e3aef` (08-07 21:09). **27.73 h were charged
inside at most 12.75 h of wall clock**, on a backend the loop drives serially
under a flock. Whatever `used_hours("kaggle")` is measuring, it is not hours
consumed. (The known triple-submission incident — one 5.5 h kernel launched
three times, `LESSONS.md` — accounts for ~16.5 h of it and is fixed; the rest is
unexplained and unexplainable from the record.)

### 5.3 It has already cost a real run

`T1.02` is `ERROR` in the ledger with the message:

> `kaggle: 0.0h left, need 0.7h`

recorded 2026-08-08T22:07:50. A Tier-1 spec was denied 42 minutes of GPU by a
counter that had billed 7.46 h more than the ceiling it was clamped against.
T1.02 is still ERROR today.

### 5.4 T0.12 is green through all of this

T0.12 PASSes and asserts `starts_full`, `refuses_when_exhausted`,
`drained_to_exact_ceiling`, `weeks_isolated`, `stale_format_key_isolated` — and
it was strengthened correctly this morning after the last audit. But every one
of those properties is checked against **synthetic `charge()` calls the test
makes itself**. Nothing reconciles the meter against Kaggle. T0.12 verifies the
accountant's arithmetic and has never once looked at the account. That is the
sharper form of "an assertion made against a saturated quantity cannot fail":
here the assertions can fire, they just aren't pointed at reality.

### 5.5 What the 45.6 Kaggle hours bought

Ten GPU-backed PASSes (T0.09–T0.11, T1.01, T1.06–T1.10, T1.12) — all sound. And
the two most expensive results in the project, **T2.01 (~7.2 h across v2–v4) and
T2.02 (6.28 h)**, are now both VOID: invalidated by T0.14's dropout finding.
That invalidation was correct and is the system working. It also means roughly
**13.5 GPU-hours currently back no admissible claim**, and the re-run that would
recover them is blocked on D3.

---

## 6. Stuck decisions

### 6.1 D3 (may the loop `git push`?) is the root blocker for 36 specs — RANK 2

`gpu.py:assert_ref_is_current` refuses to build a GPU job whose HEAD is not an
ancestor of `origin/main`, correctly, because the VM clones from GitHub. So
every GPU submission needs a push, and pushing is reserved to the owner. The
chain is: **D3 → T2.01 re-run → 36 specs → all of curiosity and all of unison.**

New evidence since D3 was written: `origin/main` is at `824339a`, pushed
2026-08-09 17:46, and only **one** commit (`db9fd7b`) is unpushed. Someone is
pushing manually — so **option 3 ("keep it your call, push manually") is what is
actually in force**, and it is working better than D3's own table suggests. What
it does not do is let the loop launch a GPU job in the slot where it becomes
ready. 23.6 Kaggle hours expire 2026-08-16.

### 6.2 D2's stated cost is wrong, and I have appended the correction

D2 says: *"Cost of the status quo: none beyond the contradiction itself, since
the code already blocks."* Measured: the status quo costs **40 blocked specs**,
of which 4 (UB.15, UB.16, T2.13, T5.09) are blocked behind T2.02 — a run that
*explicitly refused to arbitrate*. Appended to `DECISIONS_NEEDED.md` with the
enumeration.

I have **not** taken a position on which way D2 should be decided. It remains
the owner's, and the loop's recommendation (block, and fix the docstring) is
defensible.

### 6.3 The stale Kaggle block is still the first thing in `DECISIONS_NEEDED.md`

Raised this morning, not struck. It still asks the owner to choose an option
that was implemented five days ago, still claims to block T0.10/T0.11 (both
PASS), and still recommends option 3 while option 1 shipped. One line from the
owner closes it. Carried over.

### 6.4 Nothing was quietly acted on without record

I checked the converse. The one owner decision made today — care verbs / "the
owner's hands" — was recorded *in* `DECISIONS_NEEDED.md` with the owner's words
quoted, and routed to `INTEGRATION_QUEUE.md` rather than implemented. Correct.
The D1 correction (evidence confounded, do not decide) was recorded rather than
used to quietly drop the question. Also correct.

---

## 7. Bakeoff hygiene

**No bad decision has been made, because no decision has been made.**

`docs/DECISIONS_RESOLVED.md` contains nine entries, every one on spec `TEST`,
every one a unit-test fixture. Flagged this morning as builder item 4; unfixed.
`bakeoff.py:48` still hardcodes `DECISIONS = .../DECISIONS_RESOLVED.md` and
`_append_decision(res)` (line 250) takes no path override, so any future
self-test pollutes the real record again.

On the primitive's own logic, which I did read: it is sound and its fixtures
prove it.
- The learning gate fires (`TEST — VOID: arms below the 3.0-sigma learning gate: weak`).
- VOID is never rendered as a verdict-with-a-winner, and `_finish` maps it to
  `Status.VOID`, never `Status.FAIL` — so `kills` cannot fire off a bakeoff that
  refused to arbitrate.
- TIEs inside the margin are labelled TIE and resolved by cost, not promoted to
  winners (`mid leads good by only 0.38 sigma (margin 1.5)`).
- An undeclared cost yields VOID rather than a free win (the `Arm.cost` lesson,
  guarded).
- A control that *clears* the gate inverts the verdict to VOID.

So: no winner chosen inside a noise margin, no VOID treated as a verdict, no
decision made without a learning gate. The hygiene is perfect and the file is
still a fiction.

---

## 8. The honest summary — are we closer to a curious humanoid?

**Yes, today, and for the first time in a way that is not just green ticks.**

The single most important thing that happened in this repo today is PG.8. Seven
playground specs — friction, water, ladder, contact audio, the noisy TV — all
PASS, all honest, and their composition was **an empty room with `nu = 0`**. The
ladder-and-apple standard was being certified in a world where nothing could
climb anything. Jack now has a body in the world he is supposed to learn: 13
bodies, 17 actuators, 1.118 m from the ladder base, referenced from gymnasium's
own asset rather than transcribed. That is a step toward the creature, not
toward the scoreboard.

T0.14, T0.15 and T0.16 are the same kind of progress one level down: they did
not add a claim, they made ~13 GPU-hours of planned re-runs *worth running*.
Without them the D1 architecture decision would have been made on numbers
contaminated by 103.6% action drift.

**And no — on the thing GOAL.md actually asks for, we did not move.** Jack has
never taken a step under his own policy. No sense has been shown to be
load-bearing. Nothing in this repository has ever been curious about anything.
Those are 0/7, 0/16 and 0/10, they are structurally unreachable behind a VOID
whose repair is blocked on a one-line owner answer, and today the response to
that blockage was to register 31 more specs — pre-registration is a virtue only
while the gap closes, and the gap widened from 63 to 85.

The scoreboard reads 51/136. The honest reading is: **the measurement machine is
now genuinely excellent and Jack is still a memory system with a body he has
never moved.** The next real milestone is not a spec — it is a push.

---

## FOR THE BUILDER

Ranked. None requires the owner. Items 3–7 are carried over from the 12:37 audit
and were not actioned; they are all record-integrity items, which is exactly the
category that slides when the science is going well.

1. **Make the Kaggle meter measure Kaggle (§5).** Three separate defects in
   `gpu.py`:
   (a) `budget.charge()` at line 404 runs unconditionally — move it inside
   `if res.ok`, or add a `failed_hours` bucket so a crashed job is visible as
   waste rather than indistinguishable from work;
   (b) a reattach (`JACK_REUSE_KERNEL`) correctly skips `afford()` and must also
   skip `charge()`, or it double-bills compute already paid for;
   (c) `afford()` gates on `est_hours` while `charge()` bills actuals, so nothing
   caps an overrun — log loudly when `used_hours` crosses
   `KAGGLE_WEEKLY_HOURS`, because right now the ceiling is silently exceeded.
   Then **extend T0.12 with a reconciliation property**: charge a known kernel,
   read Kaggle's own reported runtime for that kernel, and assert they agree
   within a stated tolerance. Everything T0.12 asserts today is checked against
   charges the test made itself. Fixture for the control: the pre-fix
   unconditional `charge()` billing a failed job.

2. **Give the ledger a provenance field (§1.1).** `status` was hand-edited from
   `FAIL` to `VOID` for T2.01 in `9b92d14`, and T2.02 was hand-restated earlier,
   while the file header says hand-editing is forbidden. The edits were right;
   the record cannot show that they were edits. Add either an `invalidated_by`
   field (spec id + reason + commit) that `run_spec` never writes, or a
   `python -m experiments.run invalidate <SPEC> --by <SPEC> --reason ...`
   subcommand so the runner remains the only writer. Then backfill T2.01 and
   T2.02 with `invalidated_by: T0.14`.

3. **Reconcile `attempt`/`history`, or admit ignorance (§1.2).** `T2.01`,
   `T2.02`, `T1.02`, `T0.05`, `T0.09` all read `attempt: 1, history: []`, which
   is false for all five (T2.01 has four versions in git). Set un-reconstructed
   entries to `attempt: null, history: null`. A wrong integer is worse than a
   null — this is the `Arm.cost` lesson in a second file.

4. **Stop `bakeoff.py` writing to the real decision record from tests (§7).**
   `_append_decision` (line 250) takes no path parameter and `DECISIONS`
   (line 48) is a module constant. Add an output-path argument, have the
   self-tests pass a temp path, and delete the nine `TEST` entries from
   `docs/DECISIONS_RESOLVED.md` — the file currently contains nothing else.

5. **Make `Spec.control` load-bearing (§1).** 19 PASSes record `control_metrics`
   while their spec declares `control=None` (ME.5, ME.8, PG.1, PG.3, PG.4,
   T0.03, T0.05–T0.07, T0.09, T0.11, T1.04, T1.06–T1.10, T2.10, T2.12). The
   science is fine — the control ran — but the declaration is the audit surface,
   and 19 false negatives make "does this spec declare a control?" unusable as a
   check. Have `run_spec` raise when `control_fn` is supplied and
   `spec.control is None`, then backfill the 19.

6. **Give T1.03 and T1.05 controls.** T1.03: a parameter deliberately detached
   from the graph that *must* be reported as orphaned. T1.05: an unfrozen
   sentinel that *must* move. Both cheap; both convert "we observed the good
   thing" into "and the measurement can see the bad thing."

7. **Re-run ME.8 at 3 seeds.** PASS at `seeds=1` whose own commit message
   records that a **seed-2 training collapse** was fixed by a GRU retain-bias
   init. The fix was never verified at the seed that motivated it. Same shape,
   lower stakes: T1.07 (`seeds=1`).

8. **NEW — teach the runner to report what is unreachable (§1.3).** `run next`
   answers "what can I do"; nothing answers "what can I never do, and why". 40
   specs are dead behind two VOIDs and the only way I could see that was to walk
   the graph myself. Add `python -m experiments.run blocked`, printing each
   unreachable spec with its **terminal** blocker (not its immediate parent) and
   a count per root. `LESSONS.md` already carries this lesson — *"periodically
   ask which specs are unreachable and why"* — as advice to humans; make it a
   command. The one-line summary it should print today is
   `T2.01=VOID blocks 36; T2.02=VOID blocks 4`.

9. **NEW — close the loop's own end-of-iteration record (§4.3).** The 17:07
   iteration committed its work and never wrote `iteration end`; 46 starts /
   45 ends across the log. Write the end line from a `trap ... EXIT` in
   `ladder_loop.sh` so a killed iteration still records that it was killed.
   "Silence is not success" applies to the loop's log too.

10. **Consider spending the next iterations on the 22 runnable specs rather than
    registering more (§3.2).** The registry grew +31 and the ledger +9 in 24 h;
    the unrun gap is 85. Cheapest runnable units with all deps PASS: **PS.01**
    (drive layer, CPU, no body), **PG.6/PG.7**, **ME.11.B–F** (the retrieval
    bakeoff — this would also be the **first real entry in
    `DECISIONS_RESOLVED.md`**, which is worth something on its own), **UB.14**
    (cross-modal prediction — the only unison spec not behind T2.01).

---

## FOR THE OWNER

1. **D3 is the whole bottleneck, and it is one line.** *May the loop `git push`
   its own commits to `origin/main`?* Everything else in this report is
   downstream of it: T2.01 cannot re-run → 36 specs stay dead → curiosity (0/7)
   and all-senses unison (0/16) cannot acquire a single passing spec. **23.6 of
   30 Kaggle hours expire 2026-08-16 and unspent free quota is not saved.**
   Note that you appear to be pushing manually already (`origin/main` is 1 commit
   behind as I write), so option 1 or 2 would mostly formalise what is happening
   — it would just let the loop launch in the slot where the work becomes ready
   instead of waiting for the next time you look.

2. **D2's cost was understated and I have corrected it in
   `DECISIONS_NEEDED.md`.** It says the status quo costs "none beyond the
   contradiction itself." It costs **40 blocked specs**. That does not tell you
   which way to decide — blocking may well be right — but you should decide it
   knowing the number.

3. **Do not decide D1 yet.** The correction already in `DECISIONS_NEEDED.md`
   stands and today's work reinforced it: T0.16 found that the shipped GPU
   kernels evaluated with dropout live (103.6% drift on an identical state), so
   the *re-run* that D1 needs was itself contaminated until this afternoon. It
   is now correct and ready. It needs D3.

4. **Credit exhaustion cost nine dead hours last night (02:07–10:07) and is
   fixed.** The loop now walks `fable → opus → sonnet` and credit-pauses with a
   4 h self-expiry instead of burning cron slots. Verified working — five
   fallbacks engaged today, all five iterations produced a PASS. No action needed
   unless you want the primary model changed. For information: `fable` ran dry
   five separate times today.

5. **Still waiting on you from this morning:** strike the "Kaggle GPU is not
   being granted" block at the top of `DECISIONS_NEEDED.md`. It asks you to
   choose between options when option 1 shipped five days ago, and it claims to
   block T0.10/T0.11, which both PASS. It is the first thing anyone reads in
   that file.

6. **The honest number.** 51 of 136 specs demonstrated — and 40 of the 51 are
   harness, primitives, memory and playground, the four branches that run free on
   this box's CPU. Every branch that needs a GPU or a policy reads zero. The
   measurement machine is in excellent shape; Jack has not moved.

*Audited by `scripts/overseer.sh` — independent of the builder loop, read-only on
all spec, test and model code. No experiment re-run, no ledger byte changed,
nothing outside `/home/opc/jackthelearner` touched.*
