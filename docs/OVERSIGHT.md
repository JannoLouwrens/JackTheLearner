# OVERSIGHT — 32nd audit, 2026-08-26 00:37 UTC

## VERDICT: DRIFTING — the builder has been dark for **twelve consecutive hourly slots**, the pace deficit is widening monotonically and will not close before the weekly reset, and on the current trajectory **29.69 of 30 free Kaggle GPU-hours expire on Aug 30 — a day before the builder is allowed to wake up**

The 31st audit could not decompose the usage meter and said so. **This audit can,
partly, and the answer changes the recommendation**: the burn that is starving
the builder is attributable to two long-lived interactive Claude sessions in
`/home/opc` — not to the loop, and not (as I suspected on the way in) to the
pace gate's own hourly `/usage` probes, which I checked and exonerated.

Four sections have **no findings**, and they go first because a clean result
honestly reported is worth as much as a dirty one:

- **§1 ledger integrity — clean.** `run verify` re-judged all **83** auditable
  PASS entries from the record alone and probed **81** controls: 0 verdicts that
  no longer re-derive, 0 gates that ignore their control, 0 controls declared but
  never run, 0 gates that could not be replayed, 0 entries unauditable, 0
  controls run but undeclared. Independently: **0 of 84 PASS rows** carry a
  `commit` that no longer resolves in git. Two PASSes have no control at all
  (`T0.01`, `T0.10`) — both existence claims, both long-declared, unchanged.
- **§2 thresholds and controls — clean. Not one threshold moved in the loosening
  direction in seven days.** Detail in §2; the single deletion in the diff is a
  reformat, and I chased it down rather than trusting the grep.
- **§5 GPU week accounting — clean, and I went in expecting a bug.** Detail in §5.
- **§7 bakeoff hygiene — clean.** Re-read, not assumed. `PS.01/J` is recorded as
  a **VOID**, not laundered into a verdict; `PS.01/J2` declares its 2.66σ margin
  and its `screen` rationale in the open; `D2` was resolved by ledger replay with
  its losing branch recorded.

The three constitutional gates all exit 0: **coverage** 0 commitments uncovered,
0 CLAIM-DEAD; **decisions** ratchet ok (3/10 undeclared — but see RANK 4, all
three are false); **champions** ratchet ok (6/8 phantom arenas, down from 8).

---

## RANK 1 — the builder is dark, the deficit is *widening*, and the arithmetic says it stays dark past the GPU expiry

**The measurement.** Every hourly slot from `2026-08-25T13:07` to
`2026-08-26T00:07` was pace-skipped. Twelve consecutive slots. The last
iteration *started* at `2026-08-25T12:07` and ended `12:23:33` — **12 h 14 m
dark** as of this audit.

| | 13:07 | 15:07 | 18:07 | 21:07 | 00:07 |
|---|---|---|---|---|---|
| `week:all models` (**the gate**) | 38% | 40% | 43% | 45% | **51%** |
| pace line `25 + ⌈65·elapsed/100⌉` | 38% | 38% | 40% | 41% | **42%** |
| **gap over the line** | **0** | **2** | **3** | **4** | **9** |
| `week:Fable` (printed, *not* the gate) | 66% | 68% | 72% | 74% | **86%** |
| builder iterations | — | — | — | — | **0** |

**The gap is monotonic and accelerating.** Burn over the full window is 1.18
pts/h; over the last six hours it is **1.33 pts/h**, against a pace line that
rises **0.387 pts/h by construction**. The 31st audit measured 1.0 pts/h six
hours earlier and correctly refused to extrapolate. It is now measured twice,
and it is going the wrong way.

**Therefore the builder cannot recover by not running** — the only lever it has.
Projecting the measured rates:

- `week:all models` reaches the hard **90% stop** at ≈ **2026-08-27 05:00 UTC**.
  At that point the loop stops entirely and *only* an owner `.usage-resumed`
  restarts it.
- The Claude weekly meter resets at ≈ **2026-08-31 05:00 UTC** (26% elapsed at
  00:07 → 124 h remaining; matches the CLI's own "resets Aug 31, 5am (UTC)").
- **Kaggle's `%U` week 34 (Sun Aug 23 – Sat Aug 29) resets Sunday Aug 30 00:00.**
  Charged so far: **0.3111 h of 30**.

**So 29.69 free GPU-hours expire ~29 hours BEFORE the builder is permitted to
run again.** This is the third consecutive week:

| week | free Kaggle GPU-h expired unspent |
|---|---|
| W32 | 8.82 of 30 |
| W33 | 22.11 of 30 |
| **W34 (in progress)** | **29.69 of 30 on current trajectory** |

**60.6 free GPU-hours lost in three weeks, on a project whose owner has ruled
free compute only** — and the trend is worsening, not converging.

### The attribution the last audit could not make

The 31st audit wrote: *"I am one of the suspects and I cannot exonerate myself."*
I can now narrow it, and I did it by measuring rather than reasoning.

**Exonerated: the pace gate's own probes.** Each skipped hour spawns **four**
`claude -p /usage` CLI sessions (`usage_gate`→`_usage_pct`, `pace_gate`→
`--week-elapsed`, `pace_gate`→`_usage_pct`, and the skip-path `--model Fable`
read). I found all four on disk, at `:07` every hour, in
`~/.claude/projects/-home-opc/`. I suspected a self-reinforcing loop — a gate
that spends budget to decide it has no budget. **It is not happening.** Each
transcript is ~2.7 KB and contains *only* a `queue-operation` and a
`<command-name>/usage</command-name>` local-command record: **no assistant turn,
no `model` field, no `usage` block.** `/usage` is handled locally by the CLI. The
probes are free. Reporting this as a finding would have been manufacturing one.

**Attributed: two interactive sessions that are not this project's.**

| session (`~/.claude/projects/-home-opc/`) | size | last write | models present |
|---|---|---|---|
| `68804e98-…fea251aad` | **30.3 MB** | 2026-08-26 00:09 | `claude-opus-5`, **`claude-fable-5`** |
| `b76c8195-…4b97bde2fbab` | 5.9 MB | 2026-08-25 23:53 | `claude-opus-5`, **`claude-fable-5`** |

Both were live throughout the blackout window (`pgrep` shows pid 3799119 running
`--effort xhigh --permission-mode bypassPermissions`). Both sit in `/home/opc`,
not `/home/opc/jackthelearner`. **Both carry Fable turns as well as Opus turns**
— which closes the loop on the second anomaly I went looking for: `week:Fable`
rose **+20 points across 11 hours in which the builder ran zero iterations**.
That spend is not the builder's, and Fable is the builder's own model.

I remain a contributor and I say so: this overseer is 4 Opus runs/day at
`37 */6`, and its own transcripts (`d8d990be`, 739 KB, 18:48) are the third
largest jackthelearner-attributable draw on the shared meter after the builder.

**The structural fault, stated plainly:** `pace_gate()` meters the **whole
account** and throttles **only the builder**. The builder is the sole consumer on
this box with a gate, so it is the only one that can starve, and it starves in
proportion to spending it does not make and cannot see.

---

## RANK 2 — an iteration closed `rc=0` on a pilot that wrote **zero bytes**, and 12 hours later its 710-line spec is still untracked

The `12:07` iteration — the last one that ran — closed `rc=0` at `12:23:33` with:

> *"The pilot is a tracked background task — I'll be re-invoked when it completes
> (no extra monitor needed; polling would be waste). … pilot running full-size on
> seed 90 (pid 1552865, ~667 MB, healthy)."*

**Every load-bearing word of that is now false:**

| claim | measured |
|---|---|
| pid 1552865 running | **process does not exist** |
| "~667 MB, healthy" | `/data/sm03_pilot_seed90.json.log` is **0 bytes**, mtime 12:21 — it never wrote one line |
| result artifact | `/data/sm03_pilot_seed90.json` **does not exist** |
| "I'll be re-invoked when it completes" | the iteration process exited at 12:23; nothing re-invoked it |

RSS of 667 MB proves a process was *resident*, not that it was *working*. A
zero-byte log 14 minutes after launch is the artifact check that would have
caught it — and it is exactly the check this project already knows to run.

**The damage, sitting in the working tree right now:**

```
 M docs/LESSONS.md                                    (+35 lines, a real lesson)
?? experiments/tests/sm_03_nose_reports_occluded.py   (710 lines, untracked)
```

`SM.03` is the successor claim spec for **smell** — one of the owner's
constitutional senses, registered *specifically* to un-CLAIM-DEAD that
commitment two iterations ago. It is the single best GPU_SHORT candidate for the
29.69 expiring Kaggle hours in RANK 1. It has been unversioned for **12.5 hours**
and is one `git clean` from gone.

**This is the second consecutive audit to find this class of failure.** The 30th
audit's verdict line was *"an iteration closed rc=0 on a pilot that was already
dead."* The 31st audit predicted this exact file could not be rescued while the
builder was gated, because the pace-skip rescue path is scoped to three files:

```sh
HARVEST_PATHS="experiments/ledger.json experiments/gpu_budget.json experiments/gpu_submissions.jsonl"
```

**The prediction was correct and the repair was never made.** The one path that
runs while the builder is gated is scoped to the one artifact class that is not
the problem.

**And this is the deadlock that makes RANK 1 self-sealing:** the only organ that
can execute a `FOR THE BUILDER` item is the builder. `pace_gate()` runs at `:07`,
*before* the iteration ever reads this file. **The loop cannot distinguish "there
is nothing urgent to do" from "there is a flagged emergency and I am not allowed
to look at it."** Every repair below is gated behind the thing it repairs.

---

## RANK 3 — `week:Fable` is at 86% and **both** gates are structurally blind to it

The builder runs on Fable (`JACK_LOOP_MODEL=fable`, crontab `7 * * * *`).
Both gates read `week:all models`:

- `usage_gate()` — stops at 90% of **all models**. Currently reads **51%**.
- `pace_gate()` — lines against **all models**. Currently reads **51%**.
- `week:Fable` is read *only* on the skip path, printed, and explicitly labelled
  **"(not the gate)"**.

Reading `all models` for the 90% stop is correct and deliberate (it is the
owner's account-level rule). But the consequence is now live rather than
theoretical: **Fable is at 86% and rising 1.8 pts/h — it reaches 100% at roughly
2026-08-26 08:00 UTC.** At that moment the builder's own model is exhausted while
both of its gates read ~55% and say *proceed*.

The failure mode is not a stop; it is an iteration that passes every gate, starts,
and then cannot run — burning a slot and, on the evidence of RANK 2, plausibly
closing `rc=0` about it. **No instrument in the loop will report "the builder's
model is exhausted."** There is no such check anywhere.

---

## RANK 4 — all three `UNDECLARED` decisions are **already answered**, and have been reported as live deadlock risks in every audit for 17 days

`decisions --check` prints `ratchet ok (3/10 undeclared)`. I read all three
instead of relaying the count. **None of them is open:**

| reported as UNDECLARED | actual state |
|---|---|
| `D3` | header line 274: `## ~~D3 — May the loop git push?~~ **ANSWERED: YES (owner, 2026-08-10)**` |
| `The owner's hands — how does a human TOUCH Jack's world?` | body: **"DECIDED 2026-08-09, same day: YES."** Owner: *"Can you also drop stuff in for him… Yes."* |
| `Was physics-first retired by argument instead of by bakeoff?` | body: **"DECIDED 2026-08-09: (a) RUN IT."** Owner: *"schedule the run after T2.01."* |

**The mechanism.** `experiments/decisions.py:99` —

```python
_SETTLED = re.compile(r"RESOLVED|off your desk|BY THE CALENDAR", re.I)
```

— and it is matched against **headers only** (`_HEADER = ^##`). The two design
forks record their owner ruling with the word **DECIDED**, *in the body*, under a
header still reading `(OPEN, …)`. `D3`'s header says **ANSWERED**. Neither token
is in `_SETTLED`, and a body ruling is not read at all.

**Why this matters more than a cosmetic miscount.** It is the complement of the
project's own scar at `LESSONS.md:2157` — *"a false positive is credit nobody
audits, because nobody goes looking for coverage they believe they already
have."* This is the other direction: **a false alarm that everybody sees, every
audit, and nobody checks.** Thirty-one audits have relayed "3 undeclared" without
opening them. The category has been trained into noise, which is exactly how a
*genuinely* unarmed decision would now slip through unnoticed.

**The repair has a trap in it, and the obvious fix is the dangerous one.** Do
**not** widen `_SETTLED` to match `ANSWER`: header line 1454 reads

```
## D1 — DO NOT ANSWER "DO WHAT THE MEASUREMENTS SAY". …
```

and `_SETTLED` closes a key if **any** surviving header matches — so that
widening would silently close **D1, the 38-spec decision**, on a header whose
entire purpose is to say the opposite. The safe repair is a document edit, not a
regex edit, and it uses the token the tool already knows. See FOR THE BUILDER B2.

**On my standing duty to arm one decision per audit:** I am not arming one, and
that is the finding rather than a skipped chore. There is no unarmed *live* fork
to arm — all eight real open decisions already carry `DECIDE:` blocks with
defaults and `decide_by: 2026-08-31`, and the three flagged entries are settled
questions. Arming a decided question would be inventing a fork that does not
exist, which is the "manufacture problems to look useful" failure this role is
told to avoid. I have instead appended the evidence to `DECISIONS_NEEDED.md` so
the ratchet can shrink 3 → 0 honestly.

---

## RANK 5 — 84 PASS, unmoved; the last **capability** PASS was five days ago

- **84 PASS / 187 registered = 44.9% demonstrated**, down from 49.1% as the
  registry grew 169 → 187 while PASS stood still.
- Last PASS of **any** kind: `T0.21`, 2026-08-25 10:14 — a *guard* re-stamp after
  the LG registration, not a capability.
- Last **claim**-kind PASS: **`T3.01` (sight), 2026-08-21 01:28 — 5.0 days ago.**
- Everything the ladder has produced since is red or void, honestly:
  `T4.02` FAIL, `T2.07` FAIL, `LC.03` VOID, `NE.01` FAIL, `BA.02` VOID,
  `DP.05` FAIL, `T2.15` FAIL.

I want to be fair about this, because a run of FAILs is not by itself a fault —
several of these are *good* results. `T2.15` localised a real defect in the
router mechanism with every VOID gate green and an NB null beating the shipped
model; `DP.05` produced a pre-registered routing decision about the *world*
rather than a dual-process claim. That is the ladder working.

The concern is the **composition**: 14 of 24 constitutional commitments have live
claim specs and **nothing passing**, including `smell`, `voice`, `balance`,
`thermal (kills)`, `shelter/building`, `proprioception`, `plasticity`,
`sleep`, `social`, and `fast/slow` (8 declared specs, 0 passing). Registry growth
is real work, but **registering a spec is not demonstrating a capability**, and
for five days the system has done only the former.

---

## §2 — thresholds and controls, over seven days: no findings

Seven days of `git log -p` over `registry.py`, `registry_expansion.py` and
`experiments/tests/`. Every numeric change is an **addition** (the LG family,
SH.02, SM.03, DP.05, T2.15). Not one threshold moved in the loosening direction,
no control was deleted or weakened, no `_check` gained an `or`, no seed count was
reduced, no assertion was removed.

**The one deletion in the diff, chased down rather than trusted:**

```diff
-         budget=Budget.GPU_SHORT, seeds=3, depends_on=["DP.00", "VO.01"],
+         budget=Budget.GPU_SHORT, seeds=3,
+         depends_on=["DP.00", "VO.01", "LG.00"],
```

That is `DP.04` **gaining** `LG.00` and being reflowed onto two lines — a
dependency *added*, tightening the graph, in the same commit as the LG
registration exactly as its own notes required since 2026-08-10. A line-oriented
grep reads it as a removal. It is not one.

Positive signals worth recording: `## do not add seeds; 27th audit B1` is a guard
holding in-tree, and `T2.15`'s registry commit shows `protocol.py`'s
`UndeclaredControl` guard **refusing a dispatch at 0.0 s with nothing spent**
because the control was not declared. The guards are load-bearing and firing.

## §3 — drift from the goal

Everything the builder worked on in its last four iterations traces to GOAL.md.
No drift found:

| work | GOAL.md sentence it serves |
|---|---|
| `LG.00/01/02/10` registered | *"strip the diary and the learned core, and his answers about his own life must COLLAPSE"* — the anti-puppet claim, cited verbatim in GOAL.md and dangling for 16 days |
| `SM.03` implemented | *"olfaction finds food, fire and decay… the sense that works when sight fails"* |
| `SH.02` registered | *"too cold kills him"* + *"he builds a shelter"* |
| `T2.15` harvest | *"every capability claimed only by an experiment that could have failed"* |
| `coverage.goal_citations()` | protects the honesty of the ladder — SYSTEM.md's fourth duty |

**The converse, which is the harder question:** `curiosity` has 12 specs and
**1 PASS**; `one brain / unison` has 21 specs and **1 PASS**; `learning by
living` is carried by the NE/XL families, none passing. These are the three
claims GOAL.md calls the thesis itself, and they remain the least demonstrated
part of the ladder. That is not new drift — it is the same standing hole, and the
five-day PASS freeze in RANK 5 means it did not narrow.

GOAL.md still cites **4 dangling spec ids** (`GEN.02`, `GEN.03`, `GEN.06`,
`GEN.09`) — registration debt, baseline shrink-only, down from 5.

## §5 — compute honesty: the accounting is sound; the *spending* is the problem

**I went in expecting an off-by-one and did not find one.** `gpu_budget.json`
books the 2026-08-25 charge under `2026-W34`, while 2026-08-25 is **ISO** week
35. That is not a bug: `gpu.py:360` uses `%U` (Sunday-start) **deliberately**,
with the reasoning recorded in-line — ISO's Monday-start *"kept charging Sunday's
runs to the exhausted week, so the tracker refused jobs for the entire first
day."* `%U` week 34 is Sun Aug 23 – Sat Aug 29, which matches Kaggle's real
reset. Correct by design.

Every GPU hour this week is accounted for: **one** charge, `0.3111 h`
(`jack-ladder-1787631708`, T2.15), and it produced a real ledger row — a FAIL
with a full write-up and a routed follow-up. **No GPU hours were spent without a
ledger entry to show for them.** `overruns: []`. The W32 opening-balance gap
(6.3849 h) is still carried honestly in the over-stating direction.

The waste here is not misaccounted spend. It is **29.69 h of free quota that will
be spent on nothing at all** — see RANK 1.

## §6 — stuck decisions

Eight open decisions, **all armed**, all `decide_by: 2026-08-31`, none overdue,
none `MEANS-ESCALATED`. The D1 disease is not recurring: the largest entry
(`D1`, 38 specs) carries a default that *strikes* the unconstitutional option
rather than widening what is permitted, which is the correct direction.

Nothing is blocked on the owner that the system could have resolved itself with a
bakeoff. The three false `UNDECLARED` entries are RANK 4.

One item to flag rather than to fix: **five of the eight defaults fire on
2026-08-31** — the same day the Claude weekly meter resets. If the builder is
still gated then (RANK 1's projection says it will be until that morning), five
pre-registered defaults fire into a loop that cannot journal them. They are all
reversible and none widens what is permitted, so this is a sequencing risk, not
an integrity risk — but it is worth the owner knowing the date is crowded.

## §8 — the honest summary

**Are we closer to a curious humanoid that climbs the ladder than yesterday? No.
We are closer to a longer list of registered intentions.**

The system's *instruments* got better this week and that is real: `coverage`
gained goal-citation checking, the LG family finally exists so the anti-puppet
claim GOAL.md calls *"the proof he is a creature and not a costume"* is
falsifiable at all, `overseer.sh` can no longer publish a green verdict from a
dead audit, and the champions ratchet shrank 8 → 6. Four whole sections of this
audit are clean and I believe them.

But the ladder itself has not moved in five days. **84 PASS.** The registry grew
169 → 187 in the same span, so the *fraction* demonstrated fell. Smell, voice,
balance, warmth, shelter, sleep, plasticity and fast/slow all have live claim
specs and nothing passing. And the one spec written this week that could have
changed that — `SM.03`, aimed straight at a constitutional sense, with 29.69 free
GPU-hours sitting there to run it on — is **untracked in the working tree**,
its pilot dead without writing a byte, while the loop that would rescue it is
gated by a meter measuring somebody else's Opus session.

That is the whole finding in one sentence: **this project is not currently
limited by science, by compute, or by ideas. It is limited by a throttle pointed
at the wrong consumer, and by an iteration that reported "healthy" about a
process that had already produced nothing.**

---

# FOR THE BUILDER

Ordered by damage. **B1 and B2 are doable during a pace skip and must be** —
they are the only repairs that can execute while RANK 1 holds.

**B1 — rescue the orphaned unit, and widen the skip-path rescue so this cannot
recur.** Two parts, in this order:

1. Commit `experiments/tests/sm_03_nose_reports_occluded.py` (710 lines,
   untracked since 2026-08-25 12:21). Record in the commit message that its
   seed-90 pilot **produced no output**: `/data/sm03_pilot_seed90.json.log` is
   0 bytes and `/data/sm03_pilot_seed90.json` was never created, so the docstring's
   pilot numbers are **still owed** and its gates are **not frozen**. Do not
   dispatch it until a pilot actually writes a result.
2. In `scripts/ladder_loop.sh`, `harvest_bookkeeping()` currently stages only
   `HARVEST_PATHS="experiments/ledger.json experiments/gpu_budget.json
   experiments/gpu_submissions.jsonl"`. Add a second, separately-committed rescue
   of untracked/modified files under `experiments/tests/` and `docs/` — a
   mechanical "orphaned work is versioned, never judged" commit, with the torn-file
   guard already used for the three RUNNER_OUTPUTS and an explicit pathspec (the
   `add -A` ban stands). **This is the 31st audit's unrepaired item and its
   prediction came true verbatim.** Never `experiments/ledger.json` by this path —
   only the runner writes verdicts.

**B2 — stop `decisions.py` reporting three answered questions as open, without
touching the regex.** Do **not** add `ANSWER` to `_SETTLED` — header line 1454
(`## D1 — DO NOT ANSWER "DO WHAT THE MEASUREMENTS SAY"`) would then close **D1,
the 38-spec decision**, because `_SETTLED` closes a key when *any* surviving
header matches. Instead, append one **settled header** per entry using the token
the tool already reads, quoting the ruling that already exists in the body:

```
## The owner's hands — RESOLVED 2026-08-09 (owner: "Can you also drop stuff in for him… Yes.")
## Physics-first — RESOLVED 2026-08-09 (owner: "schedule the run after T2.01" — option (a), RUN IT)
## D3 — RESOLVED 2026-08-10 (owner: YES, the loop may git push)
```

Zero blast radius, no detector widened, ratchet shrinks 3 → 0. The evidence is
already appended to `DECISIONS_NEEDED.md` under the 32nd-audit housekeeping
heading. *If* you also want the regex to learn `DECIDED`, add it as a separate
commit with a fixture case proving a `DO NOT ANSWER`-style header does **not**
close its key — `_fixture()` already has the `D90`–`D95` shape to extend.

**B3 — an artifact check on every detached launch, because "healthy" was wrong
twice.** A background process is not evidence; its output is. After launching a
detached pilot, sleep ~10–15 s and assert the log file is **non-empty** before
reporting the launch succeeded; if it is 0 bytes, the launch failed at import and
must be reported as a failure in the same iteration. RSS is not liveness — the
SM.03 pilot was 667 MB resident and wrote nothing. Second occurrence in three
audits (30th audit: *"closed rc=0 on a pilot that was already dead"*).

**B4 — make the builder's own exhaustion visible (RANK 3).** `week:Fable` is at
86% and reaches 100% around 2026-08-26 08:00, while both gates read `all models`
at ~51% and say proceed. Add a **pre-flight check, not a new limit**: if the loop
model's own weekly line is ≥ 95%, log
`ABORT: builder model <M> exhausted (<n>%) — the gate reads 'all models' (<m>%)`
and exit 0 without consuming the slot. This does not weaken the 90% stop (which
stays on `all models`, per the owner's rule) and cannot let anything through that
the stop refuses — it only refuses *more*. The scar it prevents is an iteration
that passes both gates and then cannot run.

**B5 — B1's twin, on this audit's own commit.** `docs/LESSONS.md` carried 35
lines of uncommitted work — a genuinely good `[s]`-tier lesson — stranded since
12:21. I have committed it as part of this audit and attributed it to the 12:07
iteration in the commit message, because leaving a real lesson unversioned for
another five days was the worse option. Check I did not mangle it.

---

# FOR THE OWNER

**One decision, and it is the only thing standing between this project and 29.69
free GPU-hours.**

Your builder has not run since **2026-08-25 12:23**. It was not stopped by an
error, by a failing test, or by running out of ideas. It was stopped by a pacing
gate that measures **the whole Claude account** and throttles **only the
builder** — and the spend that tripped it is, on the evidence, largely your own
two interactive sessions in `/home/opc` (30.3 MB and 5.9 MB, both live through
the blackout, both carrying Opus *and* Fable turns). The builder is the only
consumer on this box with a gate, so it is the only one that starves.

The arithmetic, so you can check it:

- `week:all models` is at **51%** and rising **1.33 pts/h**; the pace line is at
  **42%** and rises **0.387 pts/h**. The gap has widened from 0 to 9 points in
  eleven hours and **cannot close by itself.**
- The loop hits the hard 90% stop around **Aug 27 05:00 UTC**.
- Your Claude week resets **Aug 31 ~05:00 UTC**.
- **Kaggle's free 30 h resets Sunday Aug 30 00:00 — one day earlier.**
- Used this week: **0.31 h of 30.**

So without an intervention, **29.69 free GPU-hours expire unused for the third
week running** (W32: 8.82 lost, W33: 22.11 lost). That is 60.6 hours of the one
resource your rules say we may spend.

**Three levers, cheapest first — all already permitted, none weakens anything:**

1. **`.usage-resumed`** with a ceiling and an expiry. This is the mechanism you
   already have; it suspends pacing and the expiry means it cannot become a
   deletion of the limit nobody remembers making.
2. **`JACK_NO_PACE=1`** on the builder's crontab line until Aug 30 — suspends
   *pacing only*; the 90% hard stop stays fully in force.
3. **Close or pause the two `/home/opc` sessions** while the builder catches up.

I am not asking you to raise the 90% ceiling and would push back if it were
proposed — that limit is yours and it is working correctly.

**A note on my own cost, since I am part of the problem:** this overseer runs 4
Opus audits/day at `37 */6` and is the largest jackthelearner-attributable draw
after the builder itself. If you want the builder's slots back and something has
to give, **cut me to `37 */12`** (two audits/day) before cutting anything that
produces science. I would rather audit half as often than watch the ladder stand
still.

**Second item, no action needed today, just a heads-up:** five of your eight open
decisions have pre-registered defaults that fire on **2026-08-31** — the same
morning the meter resets. If the builder is still gated then, five defaults fire
into a loop that cannot journal them. All five are reversible and none widens
what is permitted, so this is a sequencing wrinkle rather than a risk. Answering
even one or two before Sunday would thin out that morning.
